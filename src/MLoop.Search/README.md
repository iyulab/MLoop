# MLoop.Search

The trainer search behind [MLoop](https://github.com/iyulab/MLoop): which trainers to try, with which
settings, judged on which rows. It works on ML.NET and knows nothing of MLoop's projects, files or
command line — a host hands it data with each column's role and receives the trials it ran.

Most users want the `mloop` CLI (`dotnet tool install -g mloop`) or `MLoop.Core`, which call this
package for you. It is versioned with them.

## What is here

- `ColumnRoles` — the role of every column a search may read: numeric, categorical, text or ignored,
  and the label.
- `TrialResult` and `ITrialSink` — one trial's outcome, and where a search reports it.
- `IMetricPolicy` — the metric a search ranks trials by and which way is better.

## License

MIT
