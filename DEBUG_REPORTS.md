# Tracker Debug Reports

Debug reports are opt-in per tracker.

1. Expand **Advanced Point Settings**.
2. Enable **tracker debug recorder**.
3. Leave **Debug history** at 120 frames, or increase it for longer pre-failure history.
4. Run batch tracking.

When the tracker first becomes `UNCERTAIN`, and again if it becomes `LOST`, the app writes a report to:

```text
<project folder>/debug_reports/tracker_<uid>_<name>_<event>_frame_<frame>_<timestamp>/
```

The app also opens a debug replay window.

Report contents:

```text
debug.json          full structured numeric data
debug.csv           tabular version of the same data
roi_frames/         search ROI seen by the tracker
likelihood_maps/    colour-likelihood heatmap for each saved frame
overlay_frames/     annotated replay images
README.txt          overlay legend
```

Coordinates in `debug.csv` and `debug.json` are full original-video coordinates, not workspace-local coordinates.

Overlay colours:

```text
green circle  accepted measured position
yellow cross  Kalman/predicted position
yellow circle Kalman gate radius
magenta dot   tiny-feature colour peak
white box     centroid window
cyan border   search ROI crop border
```

The recorder only keeps the rolling history while debug is enabled. Normal runs do not allocate or save this history.

## Temporary cleanup behavior

Debug reports are temporary tuning artifacts. The application now clears the project-local `debug_reports/` folder when a new batch run starts and again when the program closes. Save or copy a report elsewhere before running another batch if you want to keep it.

Overlay frames intentionally contain only graphical markers. Text labels were removed because they obscure very small ROIs; numeric values are shown in the replay table and stored in `debug.csv` / `debug.json`. The debug replay window displays both the overlay frame and the colour-likelihood heatmap side by side.
