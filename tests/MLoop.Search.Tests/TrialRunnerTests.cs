using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Search.Candidates;
using MLoop.Search.Tuning;

namespace MLoop.Search.Tests;

/// <summary>
/// <see cref="TrialRunner"/> — the search loop: it always returns a model from its first trial,
/// records a failed trial and goes on, is the same search twice under one seed, stops at
/// once when the caller cancels, and refits the best on the whole partition.
/// </summary>
public sealed class TrialRunnerTests
{
    private sealed class Row
    {
        public bool Label { get; set; }
        public float X { get; set; }
        public float Z { get; set; }
        public string Colour { get; set; } = "";
    }

    private sealed record Auc(string Name = "auc", bool HigherIsBetter = true) : IMetricPolicy;

    private sealed class Sink : ITrialSink
    {
        public List<TrialResult> Seen { get; } = [];
        public void Completed(TrialResult trial, string rankingMetric) => Seen.Add(trial);
    }

    private static readonly string[] Colours = ["red", "green", "blue"];

    private static (IDataView Data, ColumnRoles Roles) Data(MLContext ml, int n = 240)
    {
        var random = new Random(11);
        var rows = Enumerable.Range(0, n).Select(_ =>
        {
            var colour = Colours[random.Next(3)];
            var x = (float)random.NextDouble();
            var positive = x + (colour == "red" ? 0.4 : 0) + random.NextDouble() * 0.5 > 0.9;
            return new Row { Label = positive, X = x, Z = (float)random.NextDouble(), Colour = colour };
        }).ToList();
        var roles = new ColumnRoles("Label", [new("X", ColumnRole.Numeric), new("Z", ColumnRole.Numeric), new("Colour", ColumnRole.Categorical)]);
        return (ml.Data.LoadFromEnumerable(rows), roles);
    }

    private static SearchRequest Request(MLContext ml, TimeSpan budget, int? maxTrials = null, IReadOnlyList<TrainerCandidate>? candidates = null) => new(
        candidates ?? CandidateCatalog.Binary(name => name is "FastTree" or "LbfgsLogisticRegression"),
        new Auc(),
        scored => new Dictionary<string, double> { ["auc"] = ml.BinaryClassification.EvaluateNonCalibrated(scored).AreaUnderRocCurve },
        budget,
        new SearchSettings(Seed: 3),
        MaxTrials: maxTrials);

    [Fact]
    public void ASearch_RanksTrialsByTheMetric_AndRefitsTheBest()
    {
        var ml = new MLContext(seed: 1);
        var (data, roles) = Data(ml);
        var sink = new Sink();

        var result = new TrialRunner(Request(ml, TimeSpan.FromMinutes(5), maxTrials: 10), sink).Run(data, roles);

        Assert.Equal(10, result.Trials.Count);
        Assert.Equal(result.Trials, sink.Seen);
        Assert.Equal(result.Trials.Where(t => t.Succeeded).Max(t => t.Metrics!["auc"]), result.Best.Metrics!["auc"]);
        Assert.All(result.Trials, t => Assert.Contains(t.Trainer, new[] { "FastTree", "LbfgsLogisticRegression" }));
        Assert.Contains(result.Trials, t => t.Parameters!.Any(p => p is { Key: "CategoricalEncoding", Value: "hash" }));
        Assert.Equal(5, result.BestValidation.Folds);
        Assert.Null(result.FallbackReason);
        var auc = ml.BinaryClassification.EvaluateNonCalibrated(result.Model.Transform(data)).AreaUnderRocCurve;
        Assert.True(auc > 0.7, $"refit model auc {auc}");
    }

    [Fact]
    public void TheFirstTrial_FinishesWhateverTheBudget()
    {
        var ml = new MLContext(seed: 1);
        var (data, roles) = Data(ml);

        var result = new TrialRunner(Request(ml, TimeSpan.Zero)).Run(data, roles);

        Assert.Single(result.Trials);
        Assert.True(result.Best.Succeeded);
        Assert.Equal("FastTree", result.Best.Trainer);     // the cheapest by ECI goes first
    }

    [Fact]
    public void ASeed_IsTheSameSearch()
    {
        static string Run()
        {
            var ml = new MLContext(seed: 1);
            var (data, roles) = Data(ml);
            var result = new TrialRunner(Request(ml, TimeSpan.FromMinutes(5), maxTrials: 8)).Run(data, roles);
            return string.Join("\n", result.Trials.Select(t =>
                $"{t.Trainer} {string.Join(",", t.Parameters!.Select(p => $"{p.Key}={p.Value}"))} {t.Metrics?["auc"]:R}"));
        }

        Assert.Equal(Run(), Run());
    }

    [Fact]
    public void ASeed_IsTheSameSearch_HoweverLongItsTrialsTake()
    {
        // A busy machine slows trials unevenly. Here one run's FastTree trials take far longer than the
        // other's; if the proposer read the clock, the schema it picks after the first trial would differ.
        static string Run(TimeSpan fastTreeDelay)
        {
            var ml = new MLContext(seed: 1);
            var (data, roles) = Data(ml);
            var candidates = CandidateCatalog.Binary(name => name is "FastTree" or "LbfgsLogisticRegression")
                .Select(c => c.Name != "FastTree" ? c : c with
                {
                    Build = (context, point, columns, settings) =>
                    {
                        Thread.Sleep(fastTreeDelay);
                        return c.Build(context, point, columns, settings);
                    },
                })
                .ToList();
            var result = new TrialRunner(Request(ml, TimeSpan.FromMinutes(5), maxTrials: 8, candidates)).Run(data, roles);
            return string.Join("\n", result.Trials.Select(t =>
                $"{t.Trainer} {string.Join(",", t.Parameters!.Select(p => $"{p.Key}={p.Value}"))} {t.Metrics?["auc"]:R}"));
        }

        Assert.Equal(Run(TimeSpan.Zero), Run(TimeSpan.FromMilliseconds(400)));
    }

    /// <summary>
    /// The preprocessing steps are fitted inside each fold, on its training rows, and once more on the
    /// whole partition for the refit — never on the whole partition before a fold is scored.
    /// </summary>
    [Fact]
    public void PreprocessSteps_AreFittedOnEachFoldsTrainingRows_ThenOnTheWholePartition()
    {
        var ml = new MLContext(seed: 1);
        var (data, roles) = Data(ml);
        var recorder = new FitRecorder(ml.Transforms.NormalizeMinMax("X", "X"));

        var result = new TrialRunner(Request(ml, TimeSpan.FromMinutes(5), maxTrials: 2) with { Preprocess = recorder }).Run(data, roles);

        var total = data.GetRowCount()!.Value;
        Assert.Equal(result.Trials.Count * result.BestValidation.Folds + 1, recorder.Rows.Count);
        Assert.All(recorder.Rows.Take(recorder.Rows.Count - 1), rows => Assert.InRange(rows, 1, total - 1));
        Assert.Equal(total, recorder.Rows[^1]);
        Assert.Contains("X", result.Model.Transform(data).Schema.Select(c => c.Name));
    }

    private sealed class FitRecorder(IEstimator<ITransformer> inner) : IEstimator<ITransformer>
    {
        public List<long> Rows { get; } = [];

        public ITransformer Fit(IDataView input)
        {
            long n = 0;
            using (var cursor = input.GetRowCursor([]))
                while (cursor.MoveNext())
                    n++;
            Rows.Add(n);
            return inner.Fit(input);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema) => inner.GetOutputSchema(inputSchema);
    }

    /// <summary>
    /// A trial whose schema's last trial took longer than the time left is not started — it could only run
    /// past the deadline and be dropped. Each trial here takes about 2 s (the build sleeps) under a 6 s budget:
    /// the second starts with about 3.4 s left before the refit, a third would start with about 1.4 s left.
    /// </summary>
    [Fact]
    [Trait("Category", "Slow")]   // the count rests on wall-clock margins (≥ 0.6 s), so it measures the runner too
    public void ATrialThatCannotFinishInTheTimeLeft_IsNotStarted()
    {
        var ml = new MLContext(seed: 1);
        var (data, roles) = Data(ml);
        var builds = 0;
        var lbfgs = CandidateCatalog.Binary(name => name == "LbfgsLogisticRegression").Single();
        var slow = lbfgs with
        {
            Build = (context, point, columns, settings) =>
            {
                builds++;
                Thread.Sleep(TimeSpan.FromSeconds(2));
                return lbfgs.Build(context, point, columns, settings);
            },
        };
        // One numeric column only, so there is one schema (no hash encoding of a category).
        var numeric = new ColumnRoles("Label", [new("X", ColumnRole.Numeric), new("Z", ColumnRole.Numeric)]);

        var result = new TrialRunner(Request(ml, TimeSpan.FromSeconds(6), candidates: [slow])).Run(data, numeric);

        Assert.Equal(2, result.Trials.Count);
        Assert.Equal(2, builds);
    }

    [Fact]
    public void AFailedTrial_IsRecorded_AndTheSearchGoesOn()
    {
        var ml = new MLContext(seed: 1);
        var (data, roles) = Data(ml);
        var broken = new TrainerCandidate("Broken", new SearchSpace([SearchDimension.Integer("N", 1, 10, defaultValue: 1)]), 0.5,
            (_, _, _, _) => throw new InvalidOperationException("this trainer cannot fit"));
        var candidates = new[] { broken }.Concat(CandidateCatalog.Binary(name => name == "LbfgsLogisticRegression")).ToList();

        var result = new TrialRunner(Request(ml, TimeSpan.FromMinutes(5), maxTrials: 6, candidates)).Run(data, roles);

        var failed = Assert.Single(result.Trials.Where(t => t.Trainer == "Broken").Take(1));
        Assert.Equal("this trainer cannot fit", failed.Error);
        Assert.False(failed.Succeeded);
        Assert.Equal("LbfgsLogisticRegression", result.Best.Trainer);
    }

    [Fact]
    public void WhenEveryTrialFails_TheSearchSaysWhy()
    {
        var ml = new MLContext(seed: 1);
        var (data, roles) = Data(ml);
        var broken = new TrainerCandidate("Broken", new SearchSpace([SearchDimension.Integer("N", 1, 10)]), 1,
            (_, _, _, _) => throw new InvalidOperationException("no native library"));

        var ex = Assert.Throws<SearchFailedException>(() => new TrialRunner(Request(ml, TimeSpan.Zero, candidates: [broken])).Run(data, roles));

        Assert.Contains("Broken: no native library", ex.Message);
        Assert.NotEmpty(ex.Trials);
    }

    [Fact]
    public void TheCallersCancellation_StopsTheSearch()
    {
        var ml = new MLContext(seed: 1);
        var (data, roles) = Data(ml);
        using var cts = new CancellationTokenSource();
        var sink = new CancelAfterFirst(cts);

        Assert.ThrowsAny<OperationCanceledException>(() =>
            new TrialRunner(Request(ml, TimeSpan.FromMinutes(5)), sink).Run(data, roles, cts.Token));
        Assert.Single(sink.Seen);
    }

    private sealed class CancelAfterFirst(CancellationTokenSource cts) : ITrialSink
    {
        public List<TrialResult> Seen { get; } = [];

        public void Completed(TrialResult trial, string rankingMetric)
        {
            Seen.Add(trial);
            cts.Cancel();
        }
    }
}
