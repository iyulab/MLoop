# Embedding MLoop.Core (in-process consumers)

This guide is for applications that consume **`MLoop.Core` in-process** — referencing the
SDK assemblies directly (via NuGet or a `ProjectReference`/git-submodule) and running AutoML
training/prediction inside their own process, rather than shelling out to the `mloop` CLI.

If you only use the `mloop` CLI (`dotnet tool install -g mloop`), you do not need this guide —
see [GUIDE.md](GUIDE.md).

> **Distribution note.** MLoop's SDK packages (`MLoop.Core`, `MLoop.Extensibility`,
> `MLoop.DataStore`, `MLoop.Ops`) are **not published to NuGet.org** by policy — the `mloop`
> dotnet tool is MLoop's shipping unit. In-process consumers today reference the SDK by
> git-submodule / `ProjectReference` or a private/local NuGet feed.

---

## 1. Pruning the transitive deep-learning weight (tabular-only consumers)

Since **v0.23.0**, `MLoop.Core.dll` is **Torch/Vision-free at the IL level** (the DL task
handlers live in the separate `MLoop.Core.DeepLearning` assembly — see the
[tabular-slim split](../CHANGELOG.md)). This makes it **safe** for a tabular-only consumer to
strip the unused managed DL assemblies from a `dotnet publish --self-contained` output.

The split does **not** shrink your publish output by itself: `Microsoft.ML.AutoML`
(MLoop.Core's core engine, irremovable) still *transitively* pulls in `Microsoft.ML.TorchSharp`
and `Microsoft.ML.Vision`. You prune them by excluding those transitive assets:

```xml
<PackageReference Include="Microsoft.ML.TorchSharp" ExcludeAssets="all" />
<PackageReference Include="Microsoft.ML.Vision" ExcludeAssets="all" />
<PackageReference Include="TorchSharp" ExcludeAssets="all" />
```

Verified result: the tabular self-contained publish then contains **no**
TorchSharp/Vision/LibTorchSharp assemblies, and tabular AutoML (binary/multiclass/regression,
clustering, ranking, forecasting, time-series-anomaly, anomaly-detection, recommendation) still
runs. A consumer measured ~201 MB → ~196 MB and 5 → 0 Torch/Vision/LibTorchSharp DLLs across
its engine bundles.

> If your consumer **does** use a DL task (image-classification, text-classification,
> sentence-similarity, ner, object-detection, question-answering), do **not** apply this recipe
> — reference `MLoop.Core.DeepLearning` and keep the DL assets.

### ⚠️ Placement: put the recipe in the *publishing* project, not an intermediate library

`ExcludeAssets` on a `PackageReference` **does not flow transitively through a
`ProjectReference`.** If your topology is

```
engine.exe  →  YourApp.ML (intermediate lib)  →  MLoop.Core  →  AutoML  →  Torch
```

placing the recipe in the intermediate `YourApp.ML` library has **no effect** — the Torch DLLs
still land in `engine.exe`'s publish output. The recipe must live in **the project that is
actually published** (the executable / entry-point project). A clean pattern for multiple
publishable entry points is a shared `.props` file that each executable `<Import>`s:

```xml
<!-- Directory.Torch-Exclude.props (shared) -->
<Project>
  <ItemGroup>
    <PackageReference Include="Microsoft.ML.TorchSharp" ExcludeAssets="all" />
    <PackageReference Include="Microsoft.ML.Vision" ExcludeAssets="all" />
    <PackageReference Include="TorchSharp" ExcludeAssets="all" />
  </ItemGroup>
</Project>
```

```xml
<!-- each engine .exe .csproj -->
<Import Project="..\Directory.Torch-Exclude.props" />
```

### Non-CPM consumers: specify versions explicitly

The versionless recipe above assumes **Central Package Management** (a `Directory.Packages.props`
supplying the versions). A consumer **without** CPM will fail the versionless form with
**NU1015** ("PackageReference … does not contain a version"). Add explicit versions.

**Do not freeze a hardcoded version trio** — the DL package versions move whenever
`Microsoft.ML.AutoML` bumps. Derive the exact versions your graph actually resolves:

```bash
dotnet list <your-project>.csproj package --include-transitive | \
  grep -Ei "TorchSharp|Microsoft.ML.Vision|Microsoft.ML.TorchSharp"
```

For reference only, the transitive values under `Microsoft.ML.AutoML 0.23.0` are:

| Package | Version (AutoML 0.23.0) |
|---------|-------------------------|
| `TorchSharp` | `0.102.7` |
| `Microsoft.ML.TorchSharp` | `0.23.0` |
| `Microsoft.ML.Vision` | `5.0.0` |

```xml
<PackageReference Include="Microsoft.ML.TorchSharp" Version="0.23.0" ExcludeAssets="all" />
<PackageReference Include="Microsoft.ML.Vision" Version="5.0.0" ExcludeAssets="all" />
<PackageReference Include="TorchSharp" Version="0.102.7" ExcludeAssets="all" />
```

---

## 2. Security floor for `ProjectReference` / submodule consumers

`MLoop.Core` pins `Microsoft.Bcl.Memory` to a CVE-fixed version
(**9.0.14**, clearing CVE-2026-26127 / GHSA-73j8-2gch-69rq / NU1903) as a **direct**
`PackageReference` since **v0.23.2**. This matters specifically for `ProjectReference` /
submodule consumers:

- The vulnerable **9.0.4** enters the graph *transitively* through the ML.NET stack.
- MLoop's repo build overrides it to 9.0.14 via `CentralPackageTransitivePinningEnabled`, but
  **a CPM transitive pin is repo-build-scoped — it does not cross a `ProjectReference`
  boundary.** Before v0.23.2, a submodule consumer of `MLoop.Core` silently re-inherited the
  vulnerable 9.0.4.
- By declaring `Microsoft.Bcl.Memory` as a **direct** dependency of `MLoop.Core`, the 9.0.14
  floor is now a resolved direct dependency that **propagates to every `ProjectReference`
  consumer automatically**. NuGet consumers received it already (via the transitive-pin in the
  resolved dependency set); this closes the `ProjectReference` gap.

**Action for consumers:** on MLoop.Core **≥ v0.23.2**, you can drop any manual
`Microsoft.Bcl.Memory 9.0.14` pin you were carrying to work around this — the floor now
arrives with `MLoop.Core`. Verify with `dotnet list <project> package --vulnerable
--include-transitive` (expect: no vulnerable packages).

---

## See also
- [GUIDE.md](GUIDE.md) — CLI usage
- [ARCHITECTURE.md](ARCHITECTURE.md) — in-process Core engine design
- [ECOSYSTEM.md](ECOSYSTEM.md) — where MLoop sits in the iyulab data/ML ecosystem
