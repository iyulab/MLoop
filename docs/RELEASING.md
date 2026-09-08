# Releasing MLoop

This is the maintainer's procedure for cutting a release of the `mloop` CLI. It is short on
purpose: the steps that matter are the ones a test suite cannot do for you.

## 1. Build the binary you are about to ship

```bash
dotnet build tools/MLoop.CLI -c Release
# → tools/MLoop.CLI/bin/Release/net10.0/mloop.exe  (mloop on Linux/macOS)
```

Everything below runs against **that** binary at **that** commit. A binary built an hour ago
is a binary of a different program.

## 2. Walk the new-user path — by hand, on the real surface

Do what a first-time user does, with a dataset from this repository, and read what the tool
says. This is a procedure, not a test: no unit test reads the sentences a user reads, and no
assertion checks whether a rendered path survives a copy-paste.

```bash
M=$(pwd)/tools/MLoop.CLI/bin/Release/net10.0/mloop      # or mloop.exe
cd "$(mktemp -d)"

$M init demo --task binary-classification && cd demo
cp <repo>/examples/customer-churn/datasets/customers.csv datasets/train.csv
sed -i 's/^    label: Label$/    label: Churn/' mloop.yaml

$M train                                   # auto time budget — what a user gets by default
$M predict datasets/train.csv
$M train                                   # a second run: the replacement-promotion sentence
$M list
$M promote --best
$M status
```

Pipe each command through `| cat` (or redirect to a file) at least once — Spectre lays a
redirected stream out at 80 columns, which is where paths fold.

**What to look at** — each item has failed a release before, or was found the first time this
walk was done:

| Look at | It is wrong when |
|---|---|
| Exit codes | anything but `0` on the happy path |
| `train` duration vs. the budget it printed (`Phase 1: Probe (29s)`, `Time Limit`) | training runs minutes past its own budget — a fallback trainer without a cache checkpoint did exactly this on the churn example |
| The promotion sentence | the first promotion says anything other than "first model — nothing to compare against"; the second compares against the first |
| `Model saved to:`, `Backup:`, any path | folded across two lines in the redirected output — a user who copies it gets a path that does not exist |
| stderr on success | non-empty (`$M train 2>err.txt; test -s err.txt` should fail); consumers read stderr as the failure channel |
| Repeated prefixes or icons on a line | `ℹ️  ℹ️`, `Warning: Warning:` — two layers each adding the same decoration |
| `status` vs. `mloop.yaml` | the table states a value the file does not contain (a default rendered as if configured) |
| `--json` on a command that reads data (`info`, `analyze …`) | stdout does not parse — narration from a layer below the command can land ahead of the document, and the exit code still says 0 |
| `--json` on the commands a model makes reachable (`predict`, `evaluate`, `train`) | stdout does not parse, or parses but does not say what happened. These only run once something is trained, which is why they are here rather than in a test: the automated fixture reaches every command that works without a model, and stops where a real model begins. Run each on the walk above, parse stdout, and read the document — an outcome the terminal states in words (skipped, nothing measured, a fallback taken) that no field carries is the same defect as unparseable output, reached from the other side |

Write down what you saw. If something is wrong, it is fixed **before** the release, not noted for
the next one — the point of the walk is that this binary is the one users get.

## 3. Test suites — one project at a time

```bash
for p in tests/MLoop.Tests tests/MLoop.Core.Tests tests/MLoop.Core.DeepLearning.Tests \
         tests/MLoop.API.Tests tests/MLoop.DataStore.Tests tests/MLoop.Ops.Tests tests/MLoop.Pipeline.Tests; do
  dotnet test "$p" -c Release --nologo || break
done
```

Serially, never two at once: several suites train real AutoML experiments on a ten-second
budget, and under CPU load no trial completes — a failure that describes the machine, not the
code. The loop also keeps partial results when the whole-solution form dies under memory
pressure.

If a suite produces **no test output at all** for minutes — no passes, no failures — suspect its
build output before its code: `rm -rf tests/<project>/bin tests/<project>/obj` and run it again.
A half-updated `bin/` (a test host killed mid-run, two builds overlapping) has wedged ML.NET's
internal thread pool at the first `Fit`, with every test blocked behind it and CPU at zero.

## 4. Version and changelog

- `Directory.Build.props` carries the single version. Pre-1.0: bump **minor** for features and
  breaking changes, **patch** for fixes. No prerelease suffixes.
- `CHANGELOG.md`: turn `## [Unreleased]` into `## [x.y.z] - YYYY-MM-DD` and open a fresh
  `[Unreleased]` above it. Every user-visible change in the diff has an entry.

## 5. Push, then wait for CI on that commit

```bash
git push origin main
gh run list --branch main --limit 3          # the CI run for the commit you just pushed
```

Do not continue until **CI is green on the pushed commit**. A release pipeline once sat red for
three days because the last CI signal anyone looked at belonged to an older commit.

## 6. Publish

The `Release` workflow runs on a push that changes `Directory.Build.props`. If the version was
already on `main` — a release that failed and was then fixed — that path filter is no longer
satisfied and the push only runs CI. Trigger the release by hand:

```bash
gh workflow run release.yml --ref main
gh run watch
```

A published version is permanent: a bad one is **unlisted** on NuGet, never replaced. That is
why step 2 comes before step 5.
