# natural-drift

Does the monitoring see drift that nobody injected? A time-ordered table is cut into consecutive windows and walked in order:
the oldest fifth trains the first model, each later window is predicted with `--log`, `mloop drift` is read after it and the
model's accuracy on the window is measured with `mloop evaluate`. Then a model retrained on the oldest 60 % is compared with
production on the last window (`evaluate --against production`).

The lifecycle walk (`scripts/lifecycle-check.py`) plants its drift — it moves a column and rewrites the label — so a pass there says
the wiring flags what was planted. This one reports what a real series does; it does not assert.

```
python natural_drift_walk.py --mloop "dotnet /abs/path/mloop.dll" --table electricity.csv --label class --out result.json
```

`electricity` is OpenML data id 151 (45,312 rows, the oldest first). Needs Python 3 only. Nothing is stored in this repository.
Reading: a time-index column (a value that only grows) moves in every window, so it saturates the headline level; the per-column PSI of the
other columns is where a fall in accuracy shows.
