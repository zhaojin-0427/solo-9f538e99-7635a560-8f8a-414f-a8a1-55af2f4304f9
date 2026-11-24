"""用码时间轴 (barcode usage timeline).

The raw ``scan_records`` archive is an *overlapping* set of [start, end)
windows: several records for one barcode can cover the same instant, each
naming a product, which is exactly what makes a naïve barcode/time join fan
one arrival out into many rows.

This module keeps a *derived, non-overlapping* partition per barcode:

* every segment ``[starts_at, ends_at)`` stores the de-duplicated set of
  products owned by that barcode during the segment as a JSON array;
* moments with no owning record are gaps (``[]``); they are not stored but
  are emitted as empty-array segments by the range query, so downloads cover
  the requested range without holes;
* neighbouring segments carrying the same product set are merged.

The timeline is a cache.  It is rebuilt *inside the same transaction* as the
archive write:

* a full rebuild is only used when the cache is missing (e.g. an older
  database that already holds 用码档案);
* corrections touch only the (old ∪ new) barcode × window footprint plus the
  boundary segments that may need to be stitched back together, never the
  whole table;
* everything commits or rolls back together, so a download in progress keeps
  reading the previous complete result while a backfill is running.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from typing import Iterable


def products_at(conn: sqlite3.Connection, barcode: str, at: str) -> list[str]:
    """De-duplicated products owning ``barcode`` at instant ``at``.

    Reads the timeline, not the raw archive: the segment containing ``at``
    stores exactly the distinct covering products.  Returns ``[]`` for gaps.
    """
    row = conn.execute(
        """
        SELECT products_json
          FROM barcode_timeline
         WHERE barcode = ?
           AND starts_at <= ?
           AND ends_at > ?
        """,
        (barcode, at, at),
    ).fetchone()
    if row is None:
        return []
    return json.loads(row["products_json"])


def _products_during(conn: sqlite3.Connection, barcode: str, at: str) -> list[str]:
    """Distinct raw-archive products covering ``at`` (start inclusive)."""
    rows = conn.execute(
        """
        SELECT DISTINCT product
          FROM scan_records
         WHERE barcode = ?
           AND starts_at <= ?
           AND ends_at > ?
        """,
        (barcode, at, at),
    ).fetchall()
    # Sort in Python rather than relying on SQLite's BINARY collation, so the
    # JSON array order is stable and matches Unicode ordering everywhere.
    return sorted(row["product"] for row in rows)


def build_segments(conn: sqlite3.Connection, barcode: str, lo: str, hi: str) -> list[tuple[str, str, list[str]]]:
    """Partition ``[lo, hi)`` for one barcode from the raw archive.

    The product set only changes at some archive boundary, so each interval
    between consecutive boundaries is homogeneous.  Boundaries exactly at
    ``lo`` / ``hi`` are clamped; zero-length pieces are dropped.
    """
    if lo >= hi:
        return []
    points = {lo, hi}
    bounds = conn.execute(
        """
        SELECT starts_at, ends_at FROM scan_records
         WHERE barcode = ?
           AND starts_at < ?
           AND ends_at > ?
        """,
        (barcode, hi, lo),
    ).fetchall()
    for row in bounds:
        if lo < row["starts_at"] < hi:
            points.add(row["starts_at"])
        if lo < row["ends_at"] < hi:
            points.add(row["ends_at"])
    ordered = sorted(points)
    segments: list[tuple[str, str, list[str]]] = []
    for start, end in zip(ordered, ordered[1:]):
        if end <= start:
            continue
        products = _products_during(conn, barcode, start)
        if products:
            segments.append((start, end, products))
    return _merge(segments)


def _merge(segments: Iterable[tuple[str, str, list[str]]]) -> list[tuple[str, str, list[str]]]:
    """Fuse adjacent segments whose product sets are identical."""
    merged: list[tuple[str, str, list[str]]] = []
    # Sort on the timestamps only: lists do not define a total ordering in
    # Python 3 and must never participate in the comparison.
    for start, end, products in sorted(segments, key=lambda item: (item[0], item[1])):
        if merged and merged[-1][1] == start and merged[-1][2] == products:
            merged[-1] = (merged[-1][0], end, products)
        else:
            merged.append((start, end, products))
    return merged


def _coalesce_windows(windows: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Union overlapping or touching [lo, hi) windows into disjoint spans."""
    coalesced: list[list[str]] = []
    for lo, hi in sorted(windows):
        if lo >= hi:
            continue
        if coalesced and lo <= coalesced[-1][1]:
            if hi > coalesced[-1][1]:
                coalesced[-1][1] = hi
        else:
            coalesced.append([lo, hi])
    return [(lo, hi) for lo, hi in coalesced]


def apply_changes(conn: sqlite3.Connection, spans: dict[str, list[tuple[str, str]]]) -> None:
    """Incrementally refresh the timeline for each barcode × window.

    Windows per barcode are coalesced first, so records whose windows touch
    still produce one continuous, same-set-merged partition.  For every
    affected span the stored segments overlapping it are clipped away, fresh
    segments are built from the archive for the same span, and the pieces are
    re-stitched with their neighbours (same-set neighbours merge back
    together).  Rows outside the footprint stay byte-for-byte untouched.
    """
    for barcode, windows in spans.items():
        for lo, hi in _coalesce_windows(windows):
            overlapping = conn.execute(
                """
                SELECT starts_at, ends_at, products_json
                  FROM barcode_timeline
                 WHERE barcode = ?
                   AND starts_at < ?
                   AND ends_at > ?
                 ORDER BY starts_at
                """,
                (barcode, hi, lo),
            ).fetchall()
            candidates: list[tuple[str, str, list[str]]] = list(
                build_segments(conn, barcode, lo, hi)
            )
            # Remove the old partition of this footprint first: a correction
            # that only swaps products keeps identical boundaries, so the
            # freshly built segments would otherwise clash on the primary key.
            for row in overlapping:
                conn.execute(
                    "DELETE FROM barcode_timeline WHERE barcode = ? AND starts_at = ? AND ends_at = ?",
                    (barcode, row["starts_at"], row["ends_at"]),
                )
                products = json.loads(row["products_json"])
                if row["starts_at"] < lo:
                    candidates.append((row["starts_at"], lo, products))
                if row["ends_at"] > hi:
                    candidates.append((hi, row["ends_at"], products))
            for start, end, products in _merge(candidates):
                conn.execute(
                    """
                    INSERT INTO barcode_timeline (barcode, starts_at, ends_at, products_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (barcode, start, end, json.dumps(products, ensure_ascii=False)),
                )


def rebuild_all(conn: sqlite3.Connection) -> None:
    """Recreate the whole timeline from the raw archive.

    Only used for the initial materialisation (fresh database or an older
    database whose archive predates the timeline table).
    """
    conn.execute("DELETE FROM barcode_timeline")
    barcodes = [
        row["barcode"]
        for row in conn.execute("SELECT DISTINCT barcode FROM scan_records ORDER BY barcode").fetchall()
    ]
    for barcode in barcodes:
        extent = conn.execute(
            "SELECT MIN(starts_at) AS lo, MAX(ends_at) AS hi FROM scan_records WHERE barcode = ?",
            (barcode,),
        ).fetchone()
        if extent is None or extent["lo"] is None or extent["lo"] >= extent["hi"]:
            continue
        for start, end, products in build_segments(conn, barcode, extent["lo"], extent["hi"]):
            conn.execute(
                """
                INSERT INTO barcode_timeline (barcode, starts_at, ends_at, products_json)
                VALUES (?, ?, ?, ?)
                """,
                (barcode, start, end, json.dumps(products, ensure_ascii=False)),
            )


def spans_for_records(rows: Iterable[dict], previous: dict[str, dict]) -> dict[str, list[tuple[str, str]]]:
    """Union of [start, end) footprints for new and old versions of records."""
    spans: dict[str, set[tuple[str, str]]] = defaultdict(set)

    def add(barcode: str | None, starts_at: str | None, ends_at: str | None) -> None:
        if barcode and starts_at and ends_at and starts_at < ends_at:
            spans[barcode].add((starts_at, ends_at))

    for row in rows:
        add(row.get("barcode"), row.get("starts_at"), row.get("ends_at"))
    for old in previous.values():
        add(old.get("barcode"), old.get("starts_at"), old.get("ends_at"))
    return {barcode: sorted(windows) for barcode, windows in spans.items()}


def range_segments(
    conn: sqlite3.Connection, barcode: str, start: str, end: str
) -> list[tuple[str, str, list[str]]]:
    """Timeline covering ``[start, end)`` for download, gaps rendered as ``[]``.

    Stored segments are clipped to the requested range; every uncovered piece
    becomes an empty-array segment, so the returned segments form one
    continuous partition of the requested range (start included, end
    excluded).  Defensive merging keeps same-set neighbours together even if
    stored rows were ever adjacent.
    """
    if start >= end:
        return []
    rows = conn.execute(
        """
        SELECT starts_at, ends_at, products_json
          FROM barcode_timeline
         WHERE barcode = ?
           AND starts_at < ?
           AND ends_at > ?
         ORDER BY starts_at
        """,
        (barcode, end, start),
    ).fetchall()
    pieces: list[tuple[str, str, list[str]]] = []
    cursor = start
    for row in rows:
        seg_start = max(row["starts_at"], start)
        seg_end = min(row["ends_at"], end)
        if seg_start > cursor:
            pieces.append((cursor, seg_start, []))
        pieces.append((seg_start, seg_end, json.loads(row["products_json"])))
        cursor = seg_end
    if cursor < end:
        pieces.append((cursor, end, []))
    return _merge(pieces)
