# Performance profiling

Detailed profiling is disabled during ordinary application runs.

## Enable a profiled batch run

### Windows Command Prompt

```bat
set TRACKING_PROFILE=1
python main.py
```

### PowerShell

```powershell
$env:TRACKING_PROFILE = '1'
python main.py
```

By default, after each profiled batch the application writes a new folder in:

```text
tracking_profiles/batch_YYYYMMDD_HHMMSS/
```

The final status-bar message shows the report directory.

To select another parent folder:

```bat
set TRACKING_PROFILE_DIR=C:\temp\tracking_profiles
```

## Files in each report

- `batch_frames.csv`: one row per decoded video frame. It separates sequential decode, optional GPU upload, combined tracker processing, and Qt signal queueing.
- `tracker_frames.csv`: one row per tracker and frame. `PointFastTracker` includes detailed internal stages.
- `summary.json`: totals, means, median (`p50`), `p95`, and maximum timing for batch and tracker stages.
- `README.txt`: column reminder stored alongside the report.

## PointFastTracker timing columns

| Column | Meaning |
|---|---|
| `roi_pixels`, `roi_width`, `roi_height` | Search-region size for that frame. |
| `lab_conversion_ms` | ROI BGR-to-Lab conversion. |
| `probability_map_ms` | Foreground/background Lab probability calculation. |
| `threshold_passes` | Number of segmentation thresholds attempted. |
| `threshold_morphology_ms` | Threshold-mask creation and morphological close time. |
| `connected_components_ms` | Connected-region labelling time. |
| `components_found`, `components_scored` | Candidate workload for that frame. |
| `candidate_contour_ms` | Contour extraction time. |
| `candidate_centroid_ms` | Weighted-centroid position calculation. |
| `candidate_geometry_ms` | Shape geometry calculations. |
| `candidate_colour_quality_ms` | Candidate foreground/background quality calculation. |
| `candidate_scan_total_ms` | Whole candidate search/scoring stage. |
| `point_fast_total_ms` | Total `PointFastTracker.process_frame()` time. |
| `reject_*` | Counts of candidates rejected for each reason. |

## How to interpret the first real-video report

- Large `decode_ms`: video decoding/container access is dominant.
- Large `signal_enqueue_ms` or poor responsiveness despite low worker timing: UI-thread update/rendering may remain expensive.
- Large `lab_conversion_ms` across several nearby markers: consider shared Lab conversion.
- Large `threshold_passes` and `connected_components_ms`: use a fast locked threshold with fallback only when needed.
- Large `candidate_geometry_ms` on strongly locked frames: defer geometry scoring to low-confidence or periodic audit frames.
- Large or increasing `roi_pixels`: prediction/lock uncertainty is expanding the workload.
