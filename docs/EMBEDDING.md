# Embedding MLoop.Core (in-process consumers)

This guide is for applications that consume **`MLoop.Core` in-process** — referencing the
SDK assemblies directly (via NuGet or a `ProjectReference`/git-submodule) and running AutoML
training/prediction inside their own process, rather than shelling out to the `mloop` CLI.

If you only use the `mloop` CLI (`dotnet tool install -g mloop`), you do not need this guide —
see [GUIDE.md](GUIDE.md).

> **Distribution note.** `MLoop.Core` and `MLoop.Extensibility` are published to NuGet.org at
> the CLI's version from 0.35.0: `dotnet add package MLoop.Core`. Take the package rather than a
> `ProjectReference` to the repository — a package carries its dependency floors in its nuspec,
> while the central-package pins in this repository do not cross a `ProjectReference` into your
> build. The versions `0.3.0`, `0.4.0` and `1.0.0` on NuGet.org predate the current library. A
> minimum version (`0.35.0`) resolves to the lowest match and is safe; a floating version (`*`)
> resolves to the highest and picks the old `1.0.0`. `MLoop.DataStore` and `MLoop.Ops` are not
> published.

---

## 1. The deep-learning weight (tabular-only consumers)

`MLoop.Core` carries no deep-learning dependency. Its deep-learning task handlers live in the separate
`MLoop.Core.DeepLearning` assembly, and since MLoop searches trainers with its own `MLoop.Search` the package
graph no longer reaches TorchSharp, `Microsoft.ML.TorchSharp` or `Microsoft.ML.Vision` either — they used to
arrive transitively through `Microsoft.ML.AutoML`, which MLoop no longer references. A tabular-only consumer
needs no exclusion recipe; check what your graph resolves with:

```bash
dotnet list <your-project>.csproj package --include-transitive | \
  grep -Ei "TorchSharp|Microsoft.ML.Vision|Microsoft.ML.AutoML"
```

An empty result is the expected one. If a line appears, another package of yours brings it in.

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

### The general rule, not just this package

The paragraph above is about one package, and stating it that way is what let the same failure
recur: a later transitive override was added the same way and reached no consumer either. The rule
is about the mechanism, so it is worth stating plainly.

> A central `PackageVersion` is a version decision for **this repository's own restore**. It is not
> part of the dependency graph a project hands to whoever references it. A floor declared only there
> is invisible to a `ProjectReference` consumer.

A floor therefore has to travel one of two ways, and MLoop has used both:

1. **As a direct `PackageReference`** of the library that consumers reference — the
   `Microsoft.Bcl.Memory` route above.
2. **By advancing the dependency that pulls the vulnerable version**, until that dependency's own
   manifest declares the patched floor. Then the central override is redundant and is removed rather
   than made direct — the route taken for `System.Security.Cryptography.Xml` in **v0.31.0**, once
   `FilePrepper` 0.7.4 declared `10.0.10` itself.

This is enforced rather than only documented: `SecurityFloorReachContractTests` reads
`Directory.Packages.props`, treats every entry under a `Security:` comment as a floor, and fails the
build unless that package is also a direct reference of a project under `src/`. An override written
for build coherence rather than for an advisory (the Serilog pin, so two project graphs resolve one
assembly) carries no such comment and is left alone.

---

**Action for consumers:** on MLoop.Core **≥ v0.23.2**, you can drop any manual
`Microsoft.Bcl.Memory 9.0.14` pin you were carrying to work around this — the floor now
arrives with `MLoop.Core`. Verify with `dotnet list <project> package --vulnerable
--include-transitive` (expect: no vulnerable packages).

---

## See also
- [GUIDE.md](GUIDE.md) — CLI usage
- [ARCHITECTURE.md](ARCHITECTURE.md) — in-process Core engine design
- [ECOSYSTEM.md](ECOSYSTEM.md) — where MLoop sits in the iyulab data/ML ecosystem
