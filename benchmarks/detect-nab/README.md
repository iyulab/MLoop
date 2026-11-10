# detect-nab

`mloop detect` against the Numenta Anomaly Benchmark labels (the `numenta/NAB` repository, MIT): every series in
`labels/combined_labels.json` is run with `detect --column value --json` and read with a window-hit rule — a window around each
labelled timestamp (together 10 % of the series, split over the anomalies); a window holding a flagged point is a hit, a flagged
point outside every window is a false alarm. **This is not NAB's scoring** — do not compare the numbers with published ones.

```
python detect_vs_nab.py --mloop "dotnet /abs/path/mloop.dll" --nab <NAB checkout> --out result.json
```

Needs Python 3 with pandas. `detect` does not use the ML.NET time-series native library, so this runs on any platform. Nothing is
stored in this repository.
