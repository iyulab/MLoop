# drift-sim

How `mloop drift` reads synthetic columns of several shapes - outside the product, using its own code: the profile is built with `DriftProfile.Build` and each window is read with `DriftMeasure.Compare`, so the levels printed are the ones the command reports. Synthetic data only; nothing is read from disk.

```bash
dotnet build benchmarks/drift-sim -c Release
dotnet benchmarks/drift-sim/bin/Release/net10.0/DriftSim.dll train=5000 window=300 columns=200 profiles=200 windows=5 [families=normal,poisson] [seed=1]
```

Every family fills all `columns` of the table: normal, correlated (ten-column blocks sharing one factor), heavy-tailed (t with two degrees of freedom), log-normal, Poisson, integer 0-49, and zero-inflated (60 % zeros, twice: a move of 10 and of 25 points in the zero share). Three kinds of windows are read against each profile:

- **null** - nothing moved: the share of windows where any column reads "moderate" or worse (and "significant");
- **one** - column `c0` moved by half the family's robust standard deviation: the share of windows where that column reads "moderate" or worse;
- **many** - columns `c0`..`c19` each moved by a fifth of a standard deviation: the share of windows where any column (and where a moved column) reads "moderate" or worse.

`multipliers=2,4` also reads every window as if the family error rate (0.05, the chance that a window with nothing changed reads over its bound in some column) were divided by each number, next to the reading as shipped; `multipliers=1` is the self-check (it reproduces the shipped levels). `onescale=0.6` moves the one column by 0.3 of the family's robust standard deviation instead of 0.5.

The profile is one sample: its error repeats in every window read against it, so a null rate's standard error is about `sqrt(rate * (1 - rate) / profiles)` - use many profiles with few windows each (`profiles=200 windows=5`) to resolve a rate rather than few profiles with many windows. A run of 200 profiles over all families takes about ten minutes.
