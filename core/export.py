"""
core/export.py
--------------
Exports all tracker results to a single .npz archive.

Schema
------
meta                    — JSON string (video info + all tracker configs)
{uid}_type              — str  tracker type name
{uid}_frames            — int32  (F,)  frame indices
{uid}_status            — U16    (F,)  status strings
{uid}_center            — float32 (F,2)  [row,col]   point trackers
{uid}_polygon           — float32 (F,P,2)             blob_simple
{uid}_mask              — uint8   (F,H,W)              blob_complex / color_area
{uid}_spline            — float32 (F,K,2)              curve

Polygon arrays are zero-padded to the max polygon length across all frames.
Mask arrays may be large — for 4K video with many frames they can be GBs;
the user is warned before export if estimated size > 1 GB.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from core.tracker_manager import TrackerManager
from core.video_reader import VideoReader
from tracking.base_tracker import FrameResult, TrackerType


def _export_transform(calibration: Optional[dict]) -> tuple[float, Optional[np.ndarray], str]:
    calibration = calibration or {}
    unit = str(calibration.get('export_unit', 'px'))
    origin = calibration.get('origin_rc')
    origin_rc = None if origin is None else np.asarray(origin, dtype=np.float32).reshape(2)
    if unit == 'px':
        return 1.0, origin_rc, unit
    pixels_per_meter = calibration.get('pixels_per_meter')
    if pixels_per_meter is None or float(pixels_per_meter) <= 0.0:
        raise ValueError('Physical export requires a positive pixels-per-meter calibration.')
    meters_per_unit = {'m': 1.0, 'cm': 0.01, 'mm': 0.001}[unit]
    return 1.0 / (float(pixels_per_meter) * meters_per_unit), origin_rc, unit


def _transform_rc(values: np.ndarray, scale: float, origin_rc: Optional[np.ndarray]) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    if origin_rc is not None:
        out -= origin_rc
    return out * float(scale)


def export_npz(
    path: str | Path,
    manager: TrackerManager,
    reader: VideoReader,
    calibration: Optional[dict] = None,
) -> Dict[str, int]:
    """
    Save all tracker results to *path* as a NumPy .npz archive.

    Parameters
    ----------
    path    : Destination file path (will add .npz if missing).
    manager : TrackerManager with completed batch results.
    reader  : VideoReader for metadata.

    Returns
    -------
    summary : dict  {uid: frame_count} for each exported tracker.
    """
    path = Path(path)
    if path.suffix != '.npz':
        path = path.with_suffix('.npz')

    arrays: Dict[str, np.ndarray] = {}
    summary: Dict[str, int] = {}
    coordinate_scale, origin_rc, export_unit = _export_transform(calibration)

    # ------------------------------------------------------------------
    # Meta JSON
    # ------------------------------------------------------------------
    meta = {
        'video_path': str(reader.path),
        'fps':        reader.fps,
        'width':      reader.width,
        'height':     reader.height,
        'frame_count':reader.frame_count,
        'trackers':   {},
        'export_coordinates': {
            'unit': export_unit,
            'order': ['row_y', 'col_x'],
            'origin_rc': None if origin_rc is None else origin_rc.tolist(),
        },
    }

    for tracker in manager.all_trackers():
        uid     = tracker.uid
        results = manager.get_results(uid)
        if not results:
            continue

        meta['trackers'][uid] = tracker.config.to_dict()
        summary[uid]          = len(results)

        frames_arr  = np.array([r.frame_index for r in results], dtype=np.int32)
        status_arr  = np.array([r.status.value for r in results])
        arrays[f'{uid}_type']   = np.array(tracker.config.tracker_type.value)
        arrays[f'{uid}_frames'] = frames_arr
        arrays[f'{uid}_status'] = status_arr

        ttype = tracker.config.tracker_type

        # ---- Point trackers (center) ----
        if ttype in (TrackerType.POINT_FAST, TrackerType.POINT_ACCURATE):
            centers = np.array([
                r.center if r.center is not None
                else np.array([np.nan, np.nan], dtype=np.float32)
                for r in results
            ], dtype=np.float32)
            arrays[f'{uid}_center'] = _transform_rc(centers, coordinate_scale, origin_rc)

        # ---- Blob Simple (polygon) ----
        elif ttype == TrackerType.BLOB_SIMPLE:
            # Pad polygons to uniform length
            polys = [r.polygon for r in results]
            max_pts = max((p.shape[0] for p in polys if p is not None), default=0)
            padded = np.zeros((len(results), max_pts, 2), dtype=np.float32)
            padded[:] = np.nan
            for i, p in enumerate(polys):
                if p is not None:
                    n = min(p.shape[0], max_pts)
                    padded[i, :n] = p[:n]
            arrays[f'{uid}_polygon'] = padded

        # ---- Blob Complex / Color Area (mask) ----
        elif ttype in (TrackerType.BLOB_COMPLEX, TrackerType.COLOR_AREA):
            H, W = reader.height, reader.width
            F    = len(results)
            # Estimate size
            est_gb = F * H * W / 1024**3
            if est_gb > 1.0:
                print(f"[export] Warning: mask array for {uid} is ~{est_gb:.1f} GB")
            masks = np.zeros((F, H, W), dtype=np.uint8)
            for i, r in enumerate(results):
                if r.mask is not None:
                    m = r.mask
                    # Handle size mismatch (if video was cropped)
                    mh, mw = m.shape
                    masks[i, :mh, :mw] = m
            arrays[f'{uid}_mask'] = masks

        # ---- Curve (spline) ----
        elif ttype == TrackerType.CURVE:
            splines = [r.spline_points for r in results]
            K = max((s.shape[0] for s in splines if s is not None), default=20)
            out = np.full((len(results), K, 2), np.nan, dtype=np.float32)
            for i, s in enumerate(splines):
                if s is not None:
                    k = min(s.shape[0], K)
                    out[i, :k] = s[:k]
            arrays[f'{uid}_spline'] = out

    arrays['meta'] = np.array(json.dumps(meta))

    np.savez_compressed(str(path), **arrays)
    print(f"[export] Saved {len(summary)} trackers → {path}")
    return summary


def estimate_export_size_mb(
    manager: TrackerManager,
    reader: VideoReader,
) -> float:
    """
    Rough estimate of uncompressed export size in MB.
    Used by the UI to warn before large exports.
    """
    total = 0
    for tracker in manager.all_trackers():
        results = manager.get_results(tracker.uid)
        F = len(results)
        if F == 0:
            continue
        ttype = tracker.config.tracker_type
        if ttype in (TrackerType.BLOB_COMPLEX, TrackerType.COLOR_AREA):
            total += F * reader.height * reader.width   # uint8
        elif ttype == TrackerType.BLOB_SIMPLE:
            total += F * 50 * 2 * 4   # 50 pts * 2 coords * float32
        elif ttype in (TrackerType.POINT_FAST, TrackerType.POINT_ACCURATE):
            total += F * 2 * 4
        elif ttype == TrackerType.CURVE:
            total += F * 20 * 2 * 4
    return total / 1024 / 1024




def _safe_excel_sheet_name(base: str, used: set[str]) -> str:
    """Return a unique Excel sheet name, max 31 chars and no invalid characters."""
    name = str(base or "Tracker")
    for ch in '[]:*?/\\':
        name = name.replace(ch, '_')
    name = name.strip() or "Tracker"
    candidate = name[:31]
    if candidate not in used:
        used.add(candidate)
        return candidate
    for idx in range(2, 1000):
        suffix = f"_{idx}"
        candidate = f"{name[:31-len(suffix)]}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise RuntimeError("Could not allocate unique Excel sheet name")

def export_xlsx(
    path: str | Path,
    manager: TrackerManager,
    reader: VideoReader,
    calibration: Optional[dict] = None,
) -> Dict[str, int]:
    """Export tracker results to an Excel workbook.

    The workbook is intended for inspection/analysis in spreadsheet software.
    Point trackers get row/col coordinate columns and confidence. Other tracker
    types are represented by frame/status plus a payload summary so the export
    remains reasonably sized.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise RuntimeError(
            "Excel export requires openpyxl. Install it with: pip install openpyxl"
        ) from exc

    path = Path(path)
    if path.suffix.lower() != '.xlsx':
        path = path.with_suffix('.xlsx')

    wb = Workbook()
    coordinate_scale, origin_rc, export_unit = _export_transform(calibration)
    summary_ws = wb.active
    summary_ws.title = 'Summary'
    used_sheet_names: set[str] = {'Summary'}
    header_fill = PatternFill('solid', fgColor='1F4E78')
    header_font = Font(color='FFFFFF', bold=True)

    summary_rows = [
        ('Video path', str(reader.path)),
        ('FPS', float(reader.fps)),
        ('Width', int(reader.width)),
        ('Height', int(reader.height)),
        ('Frame count', int(reader.frame_count)),
        ('Tracker count', len(manager.all_trackers())),
        ('Coordinate unit', export_unit),
        ('Coordinate origin (row, col)', '' if origin_rc is None else f'{origin_rc[0]:.4f}, {origin_rc[1]:.4f}'),
    ]
    for row_idx, (key, value) in enumerate(summary_rows, start=1):
        summary_ws.cell(row_idx, 1, key)
        summary_ws.cell(row_idx, 2, value)
    summary_ws.column_dimensions['A'].width = 18
    summary_ws.column_dimensions['B'].width = 80

    tracker_table_start = len(summary_rows) + 3
    tracker_headers = ['UID', 'Name', 'Type', 'Rows']
    for col_idx, header in enumerate(tracker_headers, start=1):
        cell = summary_ws.cell(tracker_table_start, col_idx, header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center')

    summary: Dict[str, int] = {}
    for tracker_index, tracker in enumerate(manager.all_trackers(), start=1):
        uid = tracker.uid
        results = manager.get_results(uid)
        if not results:
            continue
        summary[uid] = len(results)
        sheet_base = tracker.name or tracker.config.name or f"{tracker.config.tracker_type.value}_{uid}"
        ws = wb.create_sheet(_safe_excel_sheet_name(sheet_base, used_sheet_names))

        ttype = tracker.config.tracker_type
        if ttype in (TrackerType.POINT_FAST, TrackerType.POINT_ACCURATE):
            headers = [
                'frame', 'time_s', f'col_x_{export_unit}', f'row_y_{export_unit}',
                'status', 'correction_source', 'confidence',
            ]
            rows = []
            for result in results:
                if result.center is not None:
                    center = _transform_rc(np.asarray(result.center).reshape(1, 2), coordinate_scale, origin_rc)[0]
                    row_y = float(center[0])
                    col_x = float(center[1])
                else:
                    row_y = None
                    col_x = None
                rows.append([
                    int(result.frame_index),
                    float(result.frame_index) / float(reader.fps) if reader.fps else 0.0,
                    col_x,
                    row_y,
                    result.status.value,
                    getattr(result, 'correction_source', ''),
                    float(result.hessian_score),
                ])
        else:
            headers = ['frame', 'time_s', 'status', 'payload']
            rows = []
            for result in results:
                payload = ''
                if result.polygon is not None:
                    payload = f'polygon_points={len(result.polygon)}'
                elif result.mask is not None:
                    payload = f'mask_shape={tuple(result.mask.shape)}'
                elif result.spline_points is not None:
                    payload = f'spline_points={len(result.spline_points)}'
                rows.append([
                    int(result.frame_index),
                    float(result.frame_index) / float(reader.fps) if reader.fps else 0.0,
                    result.status.value,
                    payload,
                ])

        for col_idx, header in enumerate(headers, start=1):
            cell = ws.cell(1, col_idx, header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center')
        for row_idx, row in enumerate(rows, start=2):
            for col_idx, value in enumerate(row, start=1):
                ws.cell(row_idx, col_idx, value)
        ws.freeze_panes = 'A2'
        ws.auto_filter.ref = ws.dimensions
        for col_idx, header in enumerate(headers, start=1):
            letter = get_column_letter(col_idx)
            if header == 'status':
                ws.column_dimensions[letter].width = 14
            elif header == 'payload':
                ws.column_dimensions[letter].width = 32
            else:
                ws.column_dimensions[letter].width = 14
        # Number formats
        for row_idx in range(2, len(rows) + 2):
            ws.cell(row_idx, 2).number_format = '0.000000'
            if ttype in (TrackerType.POINT_FAST, TrackerType.POINT_ACCURATE):
                ws.cell(row_idx, 3).number_format = '0.000'
                ws.cell(row_idx, 4).number_format = '0.000'
                ws.cell(row_idx, 7).number_format = '0.000'

        summary_row = tracker_table_start + tracker_index
        summary_ws.cell(summary_row, 1, uid)
        summary_ws.cell(summary_row, 2, tracker.name)
        summary_ws.cell(summary_row, 3, tracker.config.tracker_type.value)
        summary_ws.cell(summary_row, 4, len(results))

    wb.save(path)
    print(f"[export] Saved Excel workbook → {path}")
    return summary


def export_csv(
    path: str | Path,
    manager: TrackerManager,
    reader: VideoReader,
    calibration: Optional[dict] = None,
) -> Dict[str, int]:
    """Export all tracker results as one analysis-friendly long-format CSV."""
    path = Path(path)
    if path.suffix.lower() != '.csv':
        path = path.with_suffix('.csv')
    coordinate_scale, origin_rc, export_unit = _export_transform(calibration)
    summary: Dict[str, int] = {}
    headers = (
        'tracker_uid', 'tracker_name', 'tracker_type', 'frame', 'time_s', 'status',
        f'col_x_{export_unit}', f'row_y_{export_unit}', 'confidence', 'correction_source', 'payload',
    )
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for tracker in manager.all_trackers():
            results = manager.get_results(tracker.uid)
            if not results:
                continue
            summary[tracker.uid] = len(results)
            for result in results:
                col_x = row_y = ''
                if result.center is not None:
                    center = _transform_rc(np.asarray(result.center).reshape(1, 2), coordinate_scale, origin_rc)[0]
                    row_y, col_x = float(center[0]), float(center[1])
                payload = ''
                if result.polygon is not None:
                    payload = f'polygon_points={len(result.polygon)}'
                elif result.mask is not None:
                    payload = f'mask_shape={tuple(result.mask.shape)}'
                elif result.spline_points is not None:
                    payload = f'spline_points={len(result.spline_points)}'
                writer.writerow((
                    tracker.uid, tracker.name or tracker.config.name, tracker.config.tracker_type.value,
                    int(result.frame_index),
                    float(result.frame_index) / float(reader.fps) if reader.fps else 0.0,
                    result.status.value, col_x, row_y, float(result.hessian_score),
                    getattr(result, 'correction_source', ''), payload,
                ))
    print(f"[export] Saved CSV → {path}")
    return summary
