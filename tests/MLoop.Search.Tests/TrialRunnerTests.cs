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
        // A schema is a trainer: the categories' encoding follows from their counts, it is not searched.
        Assert.All(result.Trials, t => Assert.DoesNotContain(t.Parameters!, p => p.Key == "CategoricalEncoding"));
        Assert.Equal(5, result.BestValidation.Folds);
        Assert.Null(result.FallbackReason);
        var auc = ml.BinaryClassification.EvaluateNonCalibrated(result.Model.Transform(data)).AreaUnderRocCurve;
        Assert.True(auc > 0.7, $"refit model auc {auc}");
    }

    [Fact]
    public void KeptOutOfFoldAnswers_AreTheBestTrialsOnEveryRowOnce()
    {
        var ml = new MLContext(seed: 1);
        var (data, roles) = Data(ml);

        var result = new TrialRunner(Request(ml, TimeSpan.FromMinutes(5), maxTrials: 6) with { KeepOutOfFold = true }).Run(data, roles);

        // Five folds validate every row of the partition once, each by the model fitted without its fold.
        var outOfFold = Assert.IsType<MLoop.Search.Validation.OutOfFoldScores>(result.OutOfFold);
        Assert.Equal(240, outOfFold.Count);
        Assert.Equal(240, outOfFold.Label.Count(l => l) + outOfFold.Label.Count(l => !l));
        Assert.All(outOfFold.Weight, w => Assert.Equal(1f, w));
        // Its answers are the kept trial's: their AUC over all the rows sits with the folds' mean.
        var positives = Enumerable.Range(0, outOfFold.Count).Where(i => outOfFold.Label[i]).Select(i => outOfFold.Score[i]).ToArray();
        var negatives = Enumerable.Range(0, outOfFold.Count).Where(i => !outOfFold.Label[i]).Select(i => outOfFold.Score[i]).ToArray();
        var pooled = positives.Sum(p => negatives.Count(n => n < p) + 0.5 * negatives.Count(n => n == p)) / ((double)positives.Length * negatives.Length);
        Assert.InRange(pooled, result.Best.Metrics!["auc"] - 0.05, result.Best.Metrics!["auc"] + 0.05);
        // Both candidates give a probability.
        Assert.Equal(240, outOfFold.Probability!.Count);
    }

    [Fact]
    public void AMulticlassSearch_KeepsEachRowsClassProbabilities_AndItsTrueSlot()
    {
        var ml = new MLContext(seed: 1);
        var (data, _) = Data(ml);
        var keyed = ml.Transforms.Conversion.MapValueToKey("Class", "Colour").Fit(data).Transform(data);
        var request = new SearchRequest(
            CandidateCatalog.Multiclass(name => name == "LbfgsMaximumEntropy"),
            new Auc("macro_accuracy"),
            scored => new Dictionary<string, double> { ["macro_accuracy"] = ml.MulticlassClassification.Evaluate(scored, "Class").MacroAccuracy },
            TimeSpan.FromMinutes(5), new SearchSettings(Seed: 3), MaxTrials: 2, KeepOutOfFold: true);
        var roles = new ColumnRoles("Class", [new("X", ColumnRole.Numeric), new("Z", ColumnRole.Numeric)]);

        var outOfFold = Assert.IsType<MLoop.Search.Validation.OutOfFoldScores>(new TrialRunner(request).Run(keyed, roles).OutOfFold);

        Assert.True(outOfFold.IsMulticlass);
        Assert.Equal(240, outOfFold.Count);
        Assert.All(outOfFold.ClassProbabilities, p => Assert.Equal(3, p.Length));
        Assert.All(outOfFold.Class, c => Assert.InRange(c, 0, 2));
    }

    [Fact]
    public void OutOfFoldAnswers_AreNotKept_UnlessAsked_OrForARegression()
    {
        var ml = new MLContext(seed: 1);
        var (data, roles) = Data(ml);
        Assert.Null(new TrialRunner(Request(ml, TimeSpan.FromMinutes(5), maxTrials: 2)).Run(data, roles).OutOfFold);

        // A regression on the same rows: a Single label carries no class answer.
        var x = ml.Transforms.CopyColumns("Y", "X").Fit(data).Transform(data);
        var regression = new SearchRequest(
            CandidateCatalog.Regression(name => name == "FastTree"),
            new Auc("rmse", HigherIsBetter: false),
            scored => new Dictionary<string, double> { ["rmse"] = ml.Regression.Evaluate(scored, "Y").RootMeanSquaredError },
            TimeSpan.FromMinutes(5), new SearchSettings(Seed: 3), Stratify: false, MaxTrials: 2, KeepOutOfFold: true);
        var regressionRoles = new ColumnRoles("Y", [new("Z", ColumnRole.Numeric), new("Colour", ColumnRole.Categorical)]);
        Assert.Null(new TrialRunner(regression).Run(x, regressionRoles).OutOfFold);
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
        // Numeric columns only — one schema, the one trainer.
        var numeric = new ColumnRoles("Label", [new("X", ColumnRole.Numeric), new("Z", ColumnRole.Numeric)]);

        var result = new TrialRunner(Request(ml, TimeSpan.FromSeconds(6), candidates: [slow])).Run(data, numeric);

        Assert.Equal(2, result.Trials.Count);
        Assert.Equal(3, builds);   // the two trials, and the refit, which builds its pipeline in a context of its own
    }

    /// <summary>
    /// When the slow schema no longer fits the time left, the search goes on with the one that does, rather
    /// than end with budget unspent: the slow trainer takes about 2 s a trial, the fast one a fraction, under
    /// a 6 s budget.
    /// </summary>
    [Fact]
    [Trait("Category", "Slow")]   // rests on wall-clock margins, so it measures the runner too
    public void WhenOneSchemaNoLongerFits_TheTimeLeftGoesToOneThatDoes()
    {
        var ml = new MLContext(seed: 1);
        var (data, _) = Data(ml);
        var lbfgs = CandidateCatalog.Binary(name => name == "LbfgsLogisticRegression").Single();
        var slow = lbfgs with
        {
            Name = "Slow",
            InitialCost = 0.1,   // the cheapest by estimate, so it is drawn first
            Build = (context, point, columns, settings) =>
            {
                Thread.Sleep(TimeSpan.FromSeconds(2));
                return lbfgs.Build(context, point, columns, settings);
            },
        };
        var numeric = new ColumnRoles("Label", [new("X", ColumnRole.Numeric), new("Z", ColumnRole.Numeric)]);
        var clock = System.Diagnostics.Stopwatch.StartNew();

        var result = new TrialRunner(Request(ml, TimeSpan.FromSeconds(6), candidates: [slow, lbfgs])).Run(data, numeric);

        var lastSlow = result.Trials.ToList().FindLastIndex(t => t.Trainer == "Slow");
        Assert.True(lastSlow >= 0);
        Assert.Contains(result.Trials.Skip(lastSlow + 1), t => t.Trainer == lbfgs.Name);
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(9), $"took {clock.Elapsed.TotalSeconds:0.0} s");
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

    /// <summary>
    /// The one-standard-error rule, from a large top model: the smallest model within the margin is kept, the better
    /// score between two of one size. Under the size it starts at, or without a margin, the top trial stands.
    /// </summary>
    [Fact]
    public void ALargeTopModel_GivesWayToTheSmallestWithinOneStandardError()
    {
        var auc = new Auc();
        var big = TrialRunner.LargeModelSize;
        (double, double)[] trials = [(0.80, 16), (0.952, 2 * big), (0.948, 40), (0.949, 40), (0.930, 1)];

        Assert.Equal(3, TrialRunner.Choose(trials, top: 1, margin: 0.005, auc));   // 0.949 and 0.948 both inside; 40 < 2·big, the better of the two
        Assert.Equal(1, TrialRunner.Choose(trials, top: 1, margin: 0, auc));       // no margin, no other trial inside
        (double, double)[] small = [(0.952, big - 1), (0.951, 1)];
        Assert.Equal(0, TrialRunner.Choose(small, top: 0, margin: 0.005, auc));    // a top model under the size stands
    }

    [Fact]
    public void ForALowerIsBetterMetric_TheMarginIsAbove()
    {
        var rmse = new Auc("rmse", HigherIsBetter: false);
        (double, double)[] trials = [(1.00, 3 * TrialRunner.LargeModelSize), (1.04, 10), (1.20, 1)];

        Assert.Equal(1, TrialRunner.Choose(trials, top: 0, margin: 0.05, rmse));
    }

    [Fact]
    public void AModelsSize_IsTreesTimesLeaves_AndOneForALinearModel()
    {
        var forest = CandidateCatalog.Binary().Single(c => c.Name == "FastForest");
        var linear = CandidateCatalog.Binary().Single(c => c.Name == "LbfgsLogisticRegression");

        Assert.Equal(16, TrainerCandidate.ModelSize(forest.Space.PointAt(forest.Space.Default)));   // 4 trees × 4 leaves
        Assert.Equal(1, TrainerCandidate.ModelSize(linear.Space.PointAt(linear.Space.Default)));
    }

    private static double Size(TrialResult trial) => trial.Parameters!
        .Where(p => p.Key is "NumberOfTrees" or "NumberOfLeaves")
        .Aggregate(1.0, (size, p) => size * double.Parse(p.Value, System.Globalization.CultureInfo.InvariantCulture));
}
