# anomaly-tabular

`anomaly-detection` on public classification tables whose rarer class stands for the anomaly: trained on the features only, the same
rows scored, the score read against the held-back label (AUC, share of true anomalies among the top-k scores, rows called anomalous,
rows with no score). It reports; a detector on a table that was never an anomaly problem is a stress of the wiring and a coarse
reading of the score.

```
python anomaly_tabular.py --mloop "dotnet /abs/path/mloop.dll" --out result.json --table wilt.csv:class --table sick.csv:Class ...
```

Tables used by the first reading (OpenML data ids): wilt 40983, Satellite 40900, seismic-bumps 46956, ozone-level-8hr 1487, sick 38,
taiwanese_bankruptcy_prediction 46962. Needs Python 3 with numpy and pandas. Nothing is stored in this repository.
