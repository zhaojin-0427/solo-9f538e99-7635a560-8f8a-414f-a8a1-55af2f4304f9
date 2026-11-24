"""Tests for the derived 用码时间轴 (barcode timeline).

Covers:
* materialisation from an upload and from a pre-timeline ("旧库") database;
* overlap de-duplication, gap segments, same-set neighbour merging;
* download range semantics (start included, end excluded) and CSV shape;
* scoped incremental updates (only changed barcode × window, boundary stitch);
* barcode-side verdicts reading the timeline instead of the raw archive;
* recheck limited to arrivals inside the changed footprint;
* atomic rollback keeps the previous complete timeline on failure.
"""

from __future__ import annotations

import csv
import io
import json

import pytest

from freshcheck import service, timeline
from freshcheck.db import connect, init_db


@pytest.fixture()
def db(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


def up(db, kind, body, filename="f.csv"):
    return service.upload(kind, filename, body.encode("utf-8"), db)


def raw_timeline(db, barcode):
    with connect(db) as conn:
        rows = conn.execute(
            "SELECT starts_at, ends_at, products_json FROM barcode_timeline "
            "WHERE barcode = ? ORDER BY starts_at",
            (barcode,),
        ).fetchall()
    return [(r["starts_at"], r["ends_at"], json.loads(r["products_json"])) for r in rows]


def view(db, barcode, start, end):
    with connect(db) as conn:
        return service.timeline_view(conn, barcode, start, end)


SCANS = (
    "编号,条码,商品,UTC起,UTC止\n"
    "S1,6901,苹果,2026-01-01T00:00:00Z,2026-01-02T00:00:00Z\n"
)


def test_first_upload_materialises_timeline(db):
    up(db, "scan_records", SCANS)
    assert raw_timeline(db, "6901") == [
        (
            "2026-01-01T00:00:00.000000Z",
            "2026-01-02T00:00:00.000000Z",
            ["苹果"],
        )
    ]


def test_overlapping_records_become_deduped_product_sets(db):
    up(
        db,
        "scan_records",
        "编号,条码,商品,UTC起,UTC止\n"
        "S1,6901,苹果,2026-01-01T00:00:00Z,2026-01-02T00:00:00Z\n"
        "S2,6901,香蕉,2026-01-01T12:00:00Z,2026-01-02T12:00:00Z\n"
        "S3,6901,苹果,2026-01-01T06:00:00Z,2026-01-01T18:00:00Z\n",
    )
    assert raw_timeline(db, "6901") == [
        # [00,06) and [06,12) carry the same set and are stored merged.
        ("2026-01-01T00:00:00.000000Z", "2026-01-01T12:00:00.000000Z", ["苹果"]),
        # Apple runs until 01-02T00; banana joins at 12:00 and outlasts it.
        ("2026-01-01T12:00:00.000000Z", "2026-01-02T00:00:00.000000Z", ["苹果", "香蕉"]),
        ("2026-01-02T00:00:00.000000Z", "2026-01-02T12:00:00.000000Z", ["香蕉"]),
    ]
    # The download view over the whole extent equals the stored partition.
    downloaded = view(db, "6901", "2026-01-01T00:00:00Z", "2026-01-02T12:00:00Z")
    assert [(d["starts_at"], d["ends_at"], d["products"]) for d in downloaded] == raw_timeline(db, "6901")


def test_adjacent_segments_with_same_product_set_are_merged(db):
    up(
        db,
        "scan_records",
        "编号,条码,商品,UTC起,UTC止\n"
        "S1,6901,苹果,2026-01-01T00:00:00Z,2026-01-01T12:00:00Z\n"
        "S2,6901,苹果,2026-01-01T12:00:00Z,2026-01-02T00:00:00Z\n",
    )
    # Two touching windows naming the same single product collapse into one.
    assert raw_timeline(db, "6901") == [
        ("2026-01-01T00:00:00.000000Z", "2026-01-02T00:00:00.000000Z", ["苹果"])
    ]


def test_download_fills_gaps_with_empty_arrays_and_clips_range(db):
    up(
        db,
        "scan_records",
        "编号,条码,商品,UTC起,UTC止\n"
        "S1,6901,苹果,2026-01-01T12:00:00Z,2026-01-02T00:00:00Z\n",
    )
    segments = view(db, "6901", "2026-01-01T00:00:00Z", "2026-01-02T12:00:00Z")
    got = [(s["starts_at"], s["ends_at"], s["products"]) for s in segments]
    assert got == [
        ("2026-01-01T00:00:00.000000Z", "2026-01-01T12:00:00.000000Z", []),
        ("2026-01-01T12:00:00.000000Z", "2026-01-02T00:00:00.000000Z", ["苹果"]),
        ("2026-01-02T00:00:00.000000Z", "2026-01-02T12:00:00.000000Z", []),
    ]
    # The segments partition the whole requested range with no overlaps.
    assert segments[0]["starts_at"] == "2026-01-01T00:00:00.000000Z"
    assert segments[-1]["ends_at"] == "2026-01-02T12:00:00.000000Z"
    for earlier, later in zip(segments, segments[1:]):
        assert earlier["ends_at"] == later["starts_at"]


def test_download_start_inclusive_end_exclusive(db):
    up(db, "scan_records", SCANS)
    # Request exactly the stored segment: start included, end excluded, so the
    # clipped segment is the whole window and no gap rows are appended.
    segments = view(db, "6901", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")
    assert [(s["starts_at"], s["ends_at"], s["products"]) for s in segments] == [
        ("2026-01-01T00:00:00.000000Z", "2026-01-02T00:00:00.000000Z", ["苹果"])
    ]
    # A range wholly inside a gap is one empty-array segment.
    gap = view(db, "6901", "2026-01-03T00:00:00Z", "2026-01-03T06:00:00Z")
    assert [(s["starts_at"], s["ends_at"], s["products"]) for s in gap] == [
        ("2026-01-03T00:00:00.000000Z", "2026-01-03T06:00:00.000000Z", [])
    ]


def test_csv_renders_json_arrays_and_bom(db):
    up(db, "scan_records", SCANS)
    with connect(db) as conn:
        segments = service.timeline_view(
            conn, "6901", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"
        )
    data = service.render_timeline_csv(segments, "6901")
    assert data.startswith("﻿".encode("utf-8"))
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
    assert rows[0] == ["条码", "UTC起", "UTC止", "商品"]
    assert rows[1][0] == "6901"
    assert json.loads(rows[1][3]) == ["苹果"]

    # A gap row carries an explicit empty JSON array.
    with connect(db) as conn:
        gap_segments = service.timeline_view(
            conn, "6901", "2026-01-02T00:00:00Z", "2026-01-02T01:00:00Z"
        )
    gap_rows = list(
        csv.reader(io.StringIO(service.render_timeline_csv(gap_segments, "6901").decode("utf-8-sig")))
    )
    assert json.loads(gap_rows[1][3]) == []


def test_incremental_update_only_touches_changed_barcode_and_window(db):
    up(
        db,
        "scan_records",
        "编号,条码,商品,UTC起,UTC止\n"
        "S1,6901,苹果,2026-01-01T00:00:00Z,2026-01-02T00:00:00Z\n"
        "S2,6902,香蕉,2026-01-01T00:00:00Z,2026-01-02T00:00:00Z\n",
    )
    before_6902 = raw_timeline(db, "6902")
    # A region ending before the corrected footprint (old window started at
    # 00:00, so pick a purely outside region on the other barcode's neighbour
    # side): use the gap before ownership began, plus the other barcode.
    before_gap = view(db, "6901", "2025-12-30T00:00:00Z", "2025-12-31T00:00:00Z")

    # Correct S1 to a sub-window with a new product; S2 and the parts of 6901
    # outside the corrected window must keep their exact segment boundaries.
    up(
        db,
        "scan_records",
        "编号,条码,商品,UTC起,UTC止\n"
        "S1,6901,红苹果,2026-01-01T10:00:00Z,2026-01-01T14:00:00Z\n",
        "s2.csv",
    )
    assert raw_timeline(db, "6902") == before_6902
    assert view(db, "6901", "2025-12-30T00:00:00Z", "2025-12-31T00:00:00Z") == before_gap
    # The archived ownership outside the new (shrunk) window genuinely ends,
    # so those parts become gaps; only [10,14) stays owned, by 红苹果.
    assert raw_timeline(db, "6901") == [
        ("2026-01-01T10:00:00.000000Z", "2026-01-01T14:00:00.000000Z", ["红苹果"]),
    ]
    assert view(db, "6901", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z") == [
        {"starts_at": "2026-01-01T00:00:00.000000Z", "ends_at": "2026-01-01T10:00:00.000000Z",
         "products": [], "products_json": "[]"},
        {"starts_at": "2026-01-01T10:00:00.000000Z", "ends_at": "2026-01-01T14:00:00.000000Z",
         "products": ["红苹果"], "products_json": '["红苹果"]'},
        {"starts_at": "2026-01-01T14:00:00.000000Z", "ends_at": "2026-01-02T00:00:00.000000Z",
         "products": [], "products_json": "[]"},
    ]


def test_correction_stitches_boundary_when_product_set_matches(db):
    up(
        db,
        "scan_records",
        "编号,条码,商品,UTC起,UTC止\n"
        "S1,6901,苹果,2026-01-01T00:00:00Z,2026-01-01T12:00:00Z\n"
        "S2,6901,苹果,2026-01-01T12:00:00Z,2026-01-02T00:00:00Z\n",
    )
    assert len(raw_timeline(db, "6901")) == 1  # touching same-set windows merge
    # Extend a middle record across the same product's region: the rebuilt
    # footprint fuses with same-set neighbours back into one segment.
    up(
        db,
        "scan_records",
        "编号,条码,商品,UTC起,UTC止\n"
        "S2,6901,苹果,2026-01-01T08:00:00Z,2026-01-02T00:00:00Z\n",
        "s2.csv",
    )
    assert raw_timeline(db, "6901") == [
        ("2026-01-01T00:00:00.000000Z", "2026-01-02T00:00:00.000000Z", ["苹果"])
    ]


def test_correction_clips_neighbour_segments_at_footprint_boundary(db):
    # A segment extending past the rebuild window is clipped, not deleted.
    up(
        db,
        "scan_records",
        "编号,条码,商品,UTC起,UTC止\n"
        "S1,6901,苹果,2026-01-01T00:00:00Z,2026-01-03T00:00:00Z\n"
        "S2,6901,香蕉,2026-01-02T00:00:00Z,2026-01-02T12:00:00Z\n",
    )
    # Replace S2 with an earlier banana window; the apple segment wrapping it
    # must survive outside the old/new banana footprint.
    up(
        db,
        "scan_records",
        "编号,条码,商品,UTC起,UTC止\n"
        "S2,6901,香蕉,2026-01-01T12:00:00Z,2026-01-01T18:00:00Z\n",
        "s2.csv",
    )
    assert raw_timeline(db, "6901") == [
        ("2026-01-01T00:00:00.000000Z", "2026-01-01T12:00:00.000000Z", ["苹果"]),
        ("2026-01-01T12:00:00.000000Z", "2026-01-01T18:00:00.000000Z", ["苹果", "香蕉"]),
        ("2026-01-01T18:00:00.000000Z", "2026-01-02T00:00:00.000000Z", ["苹果"]),
        # The old banana window [01-02, 01-02T12) lost banana; the wrapping
        # apple record still owns that region on its own.
        ("2026-01-02T00:00:00.000000Z", "2026-01-02T12:00:00.000000Z", ["苹果"]),
        ("2026-01-02T12:00:00.000000Z", "2026-01-03T00:00:00.000000Z", ["苹果"]),
    ]


def test_moving_barcode_refreshes_old_and_new_timelines(db):
    up(
        db,
        "scan_records",
        "编号,条码,商品,UTC起,UTC止\n"
        "S1,6901,苹果,2026-01-01T00:00:00Z,2026-01-02T00:00:00Z\n"
        "S2,6902,香蕉,2026-01-01T00:00:00Z,2026-01-02T00:00:00Z\n",
    )
    up(
        db,
        "scan_records",
        "编号,条码,商品,UTC起,UTC止\n"
        "S1,6902,苹果,2026-01-01T00:00:00Z,2026-01-02T00:00:00Z\n",
        "s2.csv",
    )
    # Old barcode lost its only record -> no stored segments (gap on download).
    assert raw_timeline(db, "6901") == []
    assert view(db, "6901", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z") == [
        {
            "starts_at": "2026-01-01T00:00:00.000000Z",
            "ends_at": "2026-01-02T00:00:00.000000Z",
            "products": [],
            "products_json": "[]",
        }
    ]
    # New barcode now owns two products over the whole window.
    assert raw_timeline(db, "6902") == [
        ("2026-01-01T00:00:00.000000Z", "2026-01-02T00:00:00.000000Z", ["苹果", "香蕉"])
    ]


def test_barcode_verdict_reads_timeline_not_raw_archive(db, monkeypatch):
    up(db, "purchase_lines", "编号,约定条码\nP1,6901\n")
    up(db, "scan_records", SCANS)
    up(
        db,
        "arrivals",
        "编号,采购行号,扫码条码,UTC时刻\nA1,P1,6901,2026-01-01T12:00:00Z\n",
    )
    with connect(db) as conn:
        d = service.arrival_detail(conn, "A1")
    assert d["released"] is True

    # Poison the raw-archive path: verdicts must keep working because they
    # read barcode_timeline, not scan_records directly.
    def boom(*_args, **_kwargs):
        raise AssertionError("rule engine must not query the raw scan archive")

    monkeypatch.setattr(timeline, "_products_during", boom)
    with connect(db) as conn:
        d = service.arrival_detail(conn, "A1")
    assert d["released"] is True
    monkeypatch.undo()

    # A gap in the timeline means "no ownership" even though raw rows exist
    # outside that instant.
    up(
        db,
        "arrivals",
        "编号,采购行号,扫码条码,UTC时刻\nA2,P1,6901,2026-01-03T00:00:00Z\n",
        "a2.csv",
    )
    with connect(db) as conn:
        d2 = service.arrival_detail(conn, "A2")
    assert {i["reason"] for i in d2["issues"]} == {"barcode_unowned"}


def test_recheck_scoped_to_arrivals_inside_footprint(db, monkeypatch):
    up(db, "purchase_lines", "编号,约定条码\nP1,6901\n")
    up(db, "scan_records", SCANS)
    up(
        db,
        "arrivals",
        "编号,采购行号,扫码条码,UTC时刻\n"
        "IN1,P1,6901,2026-01-01T11:00:00Z\n"
        "IN2,P1,6901,2026-01-01T13:00:00Z\n"
        "OUT,P1,6901,2026-01-02T12:00:00Z\n",
    )
    seen: set[str] = set()
    from freshcheck import rules

    original = rules.recheck_arrivals

    def spy(conn, arrival_nos):
        seen.update(arrival_nos)
        return original(conn, arrival_nos)

    monkeypatch.setattr(rules, "recheck_arrivals", spy)
    up(
        db,
        "scan_records",
        "编号,条码,商品,UTC起,UTC止\n"
        "S1,6901,苹果,2026-01-01T10:00:00Z,2026-01-01T14:00:00Z\n",
        "s2.csv",
    )
    monkeypatch.undo()
    # OUT sits outside the corrected window and is never rechecked.
    assert seen == {"IN1", "IN2"}


def test_failed_upload_rolls_timeline_back(db, monkeypatch):
    up(db, "scan_records", SCANS)
    before = raw_timeline(db, "6901")

    from freshcheck import rules

    def boom(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(rules, "recheck_arrivals", boom)
    with pytest.raises(RuntimeError):
        up(
            db,
            "scan_records",
            "编号,条码,商品,UTC起,UTC止\n"
            "S1,6901,坏苹果,2026-01-01T00:00:00Z,2026-01-02T00:00:00Z\n",
            "bad.csv",
        )
    monkeypatch.undo()
    assert raw_timeline(db, "6901") == before


def test_legacy_database_archives_are_materialised_on_init(tmp_path):
    """旧库档案: a database built before the timeline existed."""
    path = str(tmp_path / "legacy.db")
    # Build the old schema (no barcode_timeline) and insert an archive row.
    conn = connect(path)
    conn.executescript(
        """
        CREATE TABLE arrivals (
            arrival_no TEXT PRIMARY KEY, po_line_no TEXT NOT NULL, barcode TEXT NOT NULL,
            arrived_at TEXT NOT NULL, batch_id INTEGER NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE purchase_lines (
            po_line_no TEXT PRIMARY KEY, agreed_barcode TEXT NOT NULL,
            batch_id INTEGER NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE scan_records (
            record_no TEXT PRIMARY KEY, barcode TEXT NOT NULL, product TEXT NOT NULL,
            starts_at TEXT NOT NULL, ends_at TEXT NOT NULL,
            batch_id INTEGER NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE arrival_issues (
            arrival_no TEXT NOT NULL, side TEXT NOT NULL, reason TEXT NOT NULL,
            ref TEXT NOT NULL, detail TEXT NOT NULL,
            PRIMARY KEY (arrival_no, reason, ref));
        CREATE TABLE upload_batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, filename TEXT NOT NULL,
            row_count INTEGER NOT NULL, received_at TEXT NOT NULL,
            status TEXT NOT NULL, note TEXT);
        INSERT INTO scan_records VALUES
            ('S1','6901','苹果','2026-01-01T00:00:00.000000Z','2026-01-02T00:00:00.000000Z',1,'2026-01-01T00:00:00.000000Z'),
            ('S2','6901','香蕉','2026-01-01T12:00:00.000000Z','2026-01-02T00:00:00.000000Z',1,'2026-01-01T00:00:00.000000Z');
        """
    )
    conn.commit()
    conn.close()

    init_db(path)  # migration must rebuild the timeline from 旧库档案
    assert raw_timeline(path, "6901") == [
        ("2026-01-01T00:00:00.000000Z", "2026-01-01T12:00:00.000000Z", ["苹果"]),
        ("2026-01-01T12:00:00.000000Z", "2026-01-02T00:00:00.000000Z", ["苹果", "香蕉"]),
    ]
    with connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    # Re-running init must not rebuild or duplicate anything.
    init_db(path)
    assert len(raw_timeline(path, "6901")) == 2


def test_invalid_range_inputs_rejected(db):
    with pytest.raises(Exception):
        service.parse_timeline_range("", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")
    with pytest.raises(Exception):
        service.parse_timeline_range("6901", "not-a-time", "2026-01-02T00:00:00Z")
    with pytest.raises(Exception):
        # end must be strictly after start
        service.parse_timeline_range("6901", "2026-01-02T00:00:00Z", "2026-01-02T00:00:00Z")


def test_window_extending_both_directions_fuses_into_one_segment(db):
    up(db, "scan_records",
       "编号,条码,商品,UTC起,UTC止\nS1,6901,苹果,2026-01-02T00:00:00Z,2026-01-03T00:00:00Z\n")
    up(db, "scan_records",
       "编号,条码,商品,UTC起,UTC止\nS1,6901,苹果,2026-01-01T00:00:00Z,2026-01-04T00:00:00Z\n",
       "s2.csv")
    assert raw_timeline(db, "6901") == [
        ("2026-01-01T00:00:00.000000Z", "2026-01-04T00:00:00.000000Z", ["苹果"])
    ]


def test_product_only_swap_inside_identical_window(db):
    up(
        db,
        "scan_records",
        "编号,条码,商品,UTC起,UTC止\n"
        "S1,6902,苹果,2026-01-01T00:00:00Z,2026-01-02T00:00:00Z\n"
        "S2,6902,香蕉,2026-01-01T00:00:00Z,2026-01-02T00:00:00Z\n",
    )
    up(
        db,
        "scan_records",
        "编号,条码,商品,UTC起,UTC止\nS2,6902,梨,2026-01-01T00:00:00Z,2026-01-02T00:00:00Z\n",
        "s3.csv",
    )
    assert raw_timeline(db, "6902") == [
        ("2026-01-01T00:00:00.000000Z", "2026-01-02T00:00:00.000000Z", ["梨", "苹果"])
    ]


def test_sub_second_boundary_in_timeline(db):
    up(
        db,
        "scan_records",
        "编号,条码,商品,UTC起,UTC止\n"
        "S1,6901,苹果,2026-01-01T00:00:00Z,2026-01-01T12:00:00.500000Z\n",
    )
    with connect(db) as conn:
        assert timeline.products_at(conn, "6901", "2026-01-01T12:00:00.499999Z") == ["苹果"]
        assert timeline.products_at(conn, "6901", "2026-01-01T12:00:00.500000Z") == []
        assert timeline.products_at(conn, "6901", "2026-01-01T00:00:00.000000Z") == ["苹果"]
