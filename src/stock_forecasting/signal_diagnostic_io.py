"""Read bounded NumPy artifacts without NumPy, pickle, models or a Python ML environment."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import mmap
import re
import struct
from datetime import date
from pathlib import Path

MAX_ROWS = 10_000_000
MAX_HEADER = 65536
MAX_JSON = 4 * 1024**2
WINDOW_BYTES = 4 * 1024**2


def read_json(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > MAX_JSON:
        raise ValueError("Diagnostic JSON is symlinked or exceeds 4 MiB")
    with path.open() as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024**2):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("Diagnostic outputs are immutable; choose a new output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".partial")
    with temporary.open("x") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


class NpyReader:
    """Map little-endian float32 or fixed-width Unicode records, never complete arrays."""

    def __init__(self, path, tail=None):
        self.path = Path(path)
        self.mapping = None
        self._file = None
        with self.path.open("rb") as handle:
            prefix = handle.read(8)
            if prefix[:6] != b"\x93NUMPY" or prefix[6:] not in (
                b"\x01\x00",
                b"\x02\x00",
                b"\x03\x00",
            ):
                raise ValueError("Unsupported NumPy header")
            width = 2 if prefix[6] == 1 else 4
            raw = handle.read(width)
            if len(raw) != width:
                raise ValueError("Truncated NumPy header")
            size = int.from_bytes(raw, "little")
            if not 0 < size <= MAX_HEADER:
                raise ValueError("NumPy header exceeds the bounded size")
            header = ast.literal_eval(
                handle.read(size).decode("utf8" if prefix[6] == 3 else "latin1")
            )
            self.offset = 8 + width + size
            self.shape = header["shape"]
            if (
                not isinstance(self.shape, tuple)
                or not self.shape
                or any(type(x) is not int or x <= 0 for x in self.shape)
                or self.shape[0] > MAX_ROWS
                or header.get("fortran_order") is not False
            ):
                raise ValueError("Invalid or unbounded NumPy shape/order")
            self.fields = {}
            if tail is not None:
                if header["descr"] != "<f4" or self.shape[1:] != tuple(tail):
                    raise ValueError("Expected little-endian float32 predictions/targets")
                self.parser = struct.Struct("<" + "f" * math.prod(tail))
                self.row_bytes = self.parser.size
            else:
                if len(self.shape) != 1 or not isinstance(header["descr"], list):
                    raise ValueError("Expected structured metadata")
                self.row_bytes = 0
                for name, dtype in header["descr"]:
                    if name in self.fields or re.fullmatch(r"<U[1-9][0-9]{0,2}", dtype) is None:
                        raise ValueError("Only unique fixed Unicode metadata fields are supported")
                    length = int(dtype[2:]) * 4
                    self.fields[name] = self.row_bytes, length
                    self.row_bytes += length
                if not {"symbol", "date", "market"} <= self.fields.keys():
                    raise ValueError("Missing metadata identity fields")
            if self.path.stat().st_size != self.offset + self.row_bytes * self.shape[0]:
                raise ValueError("Truncated or oversized NumPy payload")
            if self.row_bytes > 65536:
                raise ValueError("NumPy record exceeds the fixed mapping budget")
        self._size = self.path.stat().st_size
        self._file = self.path.open("rb")
        self._start = self._length = 0

    def _window(self, offset, length):
        if (
            self.mapping is None
            or not self._start <= offset
            or offset + length > self._start + self._length
        ):
            if self.mapping is not None:
                self.mapping.close()
            self._start = offset // WINDOW_BYTES * WINDOW_BYTES
            self._length = min(
                self._size - self._start, max(WINDOW_BYTES, offset + length - self._start)
            )
            self.mapping = mmap.mmap(
                self._file.fileno(), self._length, access=mmap.ACCESS_READ, offset=self._start
            )
        return offset - self._start

    def row(self, index):
        if not 0 <= index < self.shape[0]:
            raise IndexError(index)
        offset = self._window(self.offset + index * self.row_bytes, self.row_bytes)
        values = self.parser.unpack_from(self.mapping, offset)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Nonfinite exported predictions or labels")
        return values

    def text(self, index, field):
        if not 0 <= index < self.shape[0]:
            raise IndexError(index)
        offset, size = self.fields[field]
        start = self._window(self.offset + index * self.row_bytes + offset, size)
        return self.mapping[start : start + size].decode("utf-32-le").rstrip("\x00")

    def close(self):
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def inspect_membership(metadata):
    """Check the canonical ordered membership and retain only bounded date calendars."""
    digest = hashlib.sha256(b"[")
    calendars = {}
    for i in range(metadata.shape[0]):
        symbol, timestamp, market = (metadata.text(i, k) for k in ("symbol", "date", "market"))
        day = date.fromisoformat(timestamp[:10]).isoformat()
        if not symbol or market not in ("US", "TWSE", "TPEX"):
            raise ValueError("Invalid stock/date/market identity")
        dates = calendars.setdefault(market, set())
        dates.add(day)
        if len(dates) > 10000:
            raise ValueError("Diagnostic calendar exceeds its date limit")
        if i:
            digest.update(b",")
        digest.update(
            json.dumps([symbol, timestamp], separators=(",", ":"), ensure_ascii=False).encode()
        )
    digest.update(b"]")
    return digest.hexdigest(), {market: sorted(days) for market, days in calendars.items()}


def temporal_guard(calendars, calibration_end, score_start, horizon=14):
    """Observed market dates provide a conservative trading-session gap.

    Missing observed dates can only lengthen the real market-session gap.
    They never authorize a shorter label-maturity interval.
    """
    end, start = date.fromisoformat(calibration_end), date.fromisoformat(score_start)
    if end >= start or horizon != 14:
        raise ValueError("Require ordered validation segments and a 14-session guard")
    result = {}
    for market, days in calendars.items():
        gap = [day for day in days if calibration_end < day < score_start]
        if (
            len(gap) < horizon
            or not any(day <= calibration_end for day in days)
            or not any(day >= score_start for day in days)
        ):
            raise ValueError(
                f"Insufficient calibration/score support or 14-date guard for {market}"
            )
        result[market] = {
            "observed_gap_dates": len(gap),
            "first_gap_date": gap[0],
            "last_gap_date": gap[-1],
        }
    return result
