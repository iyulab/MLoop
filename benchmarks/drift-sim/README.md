# drift-sim

How `mloop drift` reads synthetic columns of several shapes - outside the product, using its own code: the profile is built with `DriftProfile.Build` and each window is read with `DriftMeasure.Compare`, so the levels printed are the ones the command reports. Synthetic data only; nothing is read from disk.

```bash
dotnet build benchmarks/drift-sim -c Release
dotnet benchmarks/drift-sim/bin/Release/net10.0/DriftSim.dll train=5000 window=300 columns=200 profiles=200 windows=5 [families=normal,poisson] [seed=1]
```

The `mixed` family is a table of five kinds side by side (column `j` is categorical with ten levels, normal, log-normal, Poisson or zero-inflated by `j mod 5`; a categorical column moves by lifting level 0 from 0.10).

Every other family fills all `columns` of the table: normal, correlated (ten-column blocks sharing one factor), heavy-tailed (t with two degrees of freedom), log-normal, Poisson, integer 0-49, and zero-inflated (60 % zeros, twice: a move of 10 and of 25 points in the zero share). Three kinds of windows are read against each profile:

- **null** - nothing moved: the share of windows where any column reads "moderate" or worse (and "significant");
- **one** - column `c0` moved by half the family's robust standard deviation: the share of windows where that column reads "moderate" or worse;
- **many** - columns `c0`..`c19` each moved by a fifth of a standard deviation: the share of windows where any column (and where a moved column) reads "moderate" or worse.

`multipliers=2,4` also reads every window as if the family error rate (0.05, the chance that a window with nothing changed reads over its bound in some column) were divided by each number, next to the reading as shipped; `multipliers=1` is the self-check (it reproduces the shipped levels). `onescale=0.6` moves the one column by 0.3 of the family's robust standard deviation instead of 0.5.

The profile is one sample: its error repeats in every window read against it, so a null rate's standard error is about `sqrt(rate * (1 - rate) / profiles)` - use many profiles with few windows each (`profiles=200 windows=5`) to resolve a rate rather than few profiles with many windows. A run of 200 profiles over all families takes about ten minutes.

`signal=range` reads something else: the count of window values outside the training range (above the training maximum, below the minimum) as a signal of its own, beside the PSI. For a window of `m` values drawn like the `n` training values, the chance that at least `k` lie above the training maximum is exactly `C(m, k) / C(n + m, k)` (the top `k` of the pooled `n + m` values all belong to the window), whatever the distribution; ties make it conservative. It prints, per family, how often any column fires in a window with nothing moved (family-wise: 0.05 shared over `2 x columns` tests; and per column at p < 1e-3, informational) and how often the moved column `c0` fires beside how often the PSI reads it moderate or worse as shipped. `onescale` moves `c0` as in the other mode; the `outliers` families draw 10 % / 5 % of its values three times as far out.
