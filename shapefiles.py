"""
ESRI shapefiles (.shp / .shx / .dbf), read and written without extra packages.

Supports points, lines (PolyLine) and polygons, including their Z / M and
multi-part variants (Z / M values are ignored). Polygons are written as single
closed rings, clockwise, with the last point repeating the first, the way
pyshp and GIS tools expect.
"""
import struct
from datetime import datetime

import numpy as np

SHAPE_EXTS = (".shp", ".shx", ".dbf")
SHP_KIND = {1: "point", 11: "point", 21: "point", 8: "point", 18: "point", 28: "point",
            3: "line", 13: "line", 23: "line", 5: "polygon", 15: "polygon", 25: "polygon"}
SHP_CODE = {"point": 1, "line": 3, "polygon": 5}
MULTIPOINT_TYPES = (8, 18, 28)
# preferred attribute fields for a shape's label (compared lower-case); after
# these, any field with "name" in it is used
LABEL_FIELDS = ("name", "label", "wlbwellbor", "fclname", "fldname", "prlname", "well", "wellname", "id")


def ring_area(xs, ys):
    """Signed area of a ring (shoelace formula): positive when the points run anticlockwise."""
    x, y = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def closed_ring(xs, ys):
    """A polygon's points as a shapefile ring: clockwise, last point = first point."""
    pts = [(float(x), float(y)) for x, y in zip(xs, ys)]
    if ring_area(xs, ys) > 0:
        pts.reverse()
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def _header(file_bytes, code, bbox):
    return (struct.pack(">7i", 9994, 0, 0, 0, 0, 0, file_bytes // 2)
            + struct.pack("<2i", 1000, code)
            + struct.pack("<8d", *bbox, 0.0, 0.0, 0.0, 0.0))


def _fit(text, size):
    """UTF-8 bytes of text, cut to at most size bytes without splitting a character
    (UTF-8 is what pyshp and most GIS tools expect, e.g. for "Ø")."""
    data = text.encode("utf-8")
    while len(data) > size:
        text = text[:-1]
        data = text.encode("utf-8")
    return data


def write_dbf(records, fields):
    """Attribute table. fields: [(name, "C" or "N", size)]; records: one dict per shape."""
    today = datetime.now()
    rec_len = 1 + sum(size for _, _, size in fields)
    out = [struct.pack("<4BIHH20x", 3, today.year - 1900, today.month, today.day,
                       len(records), 32 + 32 * len(fields) + 1, rec_len)]
    for name, kind, size in fields:
        out.append(struct.pack("<11sc4xBB14x", name.encode("ascii"), kind.encode("ascii"), size, 0))
    out.append(b"\r")
    for rec in records:
        row = b" "  # not deleted
        for name, kind, size in fields:
            text = _fit(str(rec.get(name, "")), size)
            row += text.rjust(size) if kind == "N" else text.ljust(size)
        out.append(row)
    out.append(b"\x1a")
    return b"".join(out)


def write_shapefile(kind, geoms, records, fields):
    """Shapefile of one kind ("point", "line" or "polygon"). geoms: one list of
    (x, y) per shape (a single point for "point"; a closed ring for "polygon",
    see closed_ring). Returns {".shp": bytes, ".shx": bytes, ".dbf": bytes}."""
    code = SHP_CODE[kind]
    shp_recs, shx_recs, offset = [], [], 100
    for i, geom in enumerate(geoms, 1):
        pts = np.asarray(geom, dtype="<f8").reshape(-1, 2)
        if kind == "point":
            content = struct.pack("<i2d", code, pts[0, 0], pts[0, 1])
        else:
            box = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
            content = struct.pack("<i4d2i", code, *box, 1, len(pts)) + struct.pack("<i", 0) + pts.tobytes()
        shp_recs.append(struct.pack(">2i", i, len(content) // 2) + content)
        shx_recs.append(struct.pack(">2i", offset // 2, len(content) // 2))
        offset += 8 + len(content)
    allpts = (np.concatenate([np.asarray(g, dtype=np.float64).reshape(-1, 2) for g in geoms])
              if geoms else np.zeros((1, 2)))
    bbox = (allpts[:, 0].min(), allpts[:, 1].min(), allpts[:, 0].max(), allpts[:, 1].max())
    shp_body, shx_body = b"".join(shp_recs), b"".join(shx_recs)
    return {
        ".shp": _header(100 + len(shp_body), code, bbox) + shp_body,
        ".shx": _header(100 + len(shx_body), code, bbox) + shx_body,
        ".dbf": write_dbf(records, fields),
    }


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_shp(data):
    """Shapes in a .shp file: (kind, shapes). Each shape is a list of parts
    (arrays of x, y): one part per point for points / multipoints, one per
    line or ring otherwise. Null records give an empty list, so the list lines
    up with the .dbf rows."""
    if len(data) < 100 or struct.unpack(">i", data[:4])[0] != 9994:
        raise ValueError("it isn't a valid .shp file")
    file_type = struct.unpack("<i", data[32:36])[0]
    kind = SHP_KIND.get(file_type)
    if kind is None:
        raise ValueError(f"shape type {file_type} isn't supported (points, lines and polygons are)")
    shapes, pos = [], 100
    while pos + 8 <= len(data):
        words = struct.unpack(">i", data[pos + 4:pos + 8])[0]
        content = data[pos + 8:pos + 8 + 2 * words]
        pos += 8 + 2 * words
        if len(content) < 4:
            break
        st = struct.unpack("<i", content[:4])[0]
        if SHP_KIND.get(st) != kind:  # null shape (0) or anything unexpected
            shapes.append([])
        elif kind == "point" and st not in MULTIPOINT_TYPES:
            shapes.append([np.frombuffer(content, dtype="<f8", count=2, offset=4).reshape(1, 2)])
        elif st in MULTIPOINT_TYPES:
            n_pts = struct.unpack("<i", content[36:40])[0]
            pts = np.frombuffer(content, dtype="<f8", count=2 * n_pts, offset=40).reshape(-1, 2)
            shapes.append([p.reshape(1, 2) for p in pts])
        else:
            n_parts, n_pts = struct.unpack("<2i", content[36:44])
            parts = list(struct.unpack(f"<{n_parts}i", content[44:44 + 4 * n_parts])) + [n_pts]
            pts = np.frombuffer(content, dtype="<f8", count=2 * n_pts, offset=44 + 4 * n_parts).reshape(-1, 2)
            shapes.append([pts[a:b] for a, b in zip(parts, parts[1:])])
    return kind, shapes


def _text(raw):
    try:
        return raw.decode("utf-8").strip()
    except UnicodeDecodeError:  # older files are usually Latin-1
        return raw.decode("latin-1").strip()


def read_dbf(data):
    """Rows of a .dbf attribute table as dicts (upper-case field names, text values)."""
    n, head_len, rec_len = struct.unpack("<IHH", data[4:12])
    fields, pos = [], 32
    while pos + 32 <= head_len and data[pos] != 0x0D:
        name = data[pos:pos + 11].split(b"\0")[0].decode("latin-1").strip().upper()
        fields.append((name, data[pos + 16]))
        pos += 32
    records = []
    for i in range(n):
        row = data[head_len + i * rec_len:head_len + (i + 1) * rec_len]
        if len(row) < rec_len:
            break
        rec, off = {}, 1
        for name, size in fields:
            rec[name] = _text(row[off:off + size])
            off += size
        records.append(rec)
    return records


def record_label(rec):
    """A short name for a shape from its attributes ("" if none fits)."""
    lower = {k.lower(): v for k, v in rec.items()}
    for key in LABEL_FIELDS:
        if lower.get(key):
            return lower[key]
    for key, value in lower.items():
        if "name" in key and value:
            return value
    return ""
