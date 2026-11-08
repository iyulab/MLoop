using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Search.Candidates;

namespace MLoop.Search.Tests;

/// <summary>
/// The model a search returns is its best trial fitted again on the whole training partition, and that refit has to
/// reproduce the score that chose the trial. A solver whose result moves with the row count breaks the assumption:
/// a linear multiclass model scored 0.92 to 0.97 on the 80-row folds of the 120-row Iris partition and 0.67 on the 120 rows it was
/// then fitted on, its own training rows. The refit is scored on the rows it learned; one that falls far short does not stand.
/// </summary>
public sealed class RefitReproducesTests
{
    private sealed class Row
    {
        [LoadColumn(0)] public float SepalLength { get; set; }
        [LoadColumn(1)] public float SepalWidth { get; set; }
        [LoadColumn(2)] public float PetalLength { get; set; }
        [LoadColumn(3)] public float PetalWidth { get; set; }
        [LoadColumn(4)] public string Species { get; set; } = "";
    }

    private sealed record Accuracy(string Name = "micro_accuracy", bool HigherIsBetter = true) : IMetricPolicy;

    [Theory]
    [InlineData(0.97, 0.67, 0.03, true, false)]   // the Iris refit: far below what the folds scored
    [InlineData(0.92, 0.90, 0.02, true, true)]    // a little under is what a model scored on its own rows may do
    [InlineData(0.92, 0.99, 0.02, true, true)]    // better on its own rows, as a fitted model usually is
    [InlineData(10.0, 14.0, 0.5, false, false)]   // an error metric: smaller is better, so larger is the shortfall
    [InlineData(10.0, 9.0, 0.5, false, true)]
    [InlineData(0.05, 0.04, 0.02, true, true)]    // a small score: the standard errors are the slack
    [InlineData(0.05, -0.2, 0.02, true, false)]
    public void ARefitThatFallsFarShortOfItsValidatedScoreDoesNotReproduceIt(
        double validated, double learned, double standardError, bool higherIsBetter, bool reproduces)
        => Assert.Equal(reproduces, TrialRunner.RefitReproduces(validated, learned, standardError, higherIsBetter));

    [Fact]
    public void ARefitWithNoScoreToJudgeByStands()
        => Assert.True(TrialRunner.RefitReproduces(0.9, null, 0.01, true));

    [Fact]
    public void ALinearMulticlassWinnerWhoseRefitCollapsesOnIris_IsReplacedByATrialWhoseRefitLearnedItsRows()
    {
        var ml = new MLContext(1);
        var path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "iris.csv");
        var data = ml.Data.ShuffleRows(ml.Data.LoadFromTextFile<Row>(path, separatorChar: ',', hasHeader: true), seed: 3);
        var split = ml.Data.TrainTestSplit(data, 0.2, seed: 42);
        var roles = new ColumnRoles("Species",
            [new("SepalLength", ColumnRole.Numeric), new("SepalWidth", ColumnRole.Numeric), new("PetalLength", ColumnRole.Numeric), new("PetalWidth", ColumnRole.Numeric)]);
        var request = new SearchRequest(
            CandidateCatalog.Multiclass(name => name is "SdcaMaximumEntropy" or "SdcaLogisticRegressionOva"),
            new Accuracy(),
            scored => new Dictionary<string, double> { ["micro_accuracy"] = ml.MulticlassClassification.Evaluate(scored, "Species").MicroAccuracy },
            TimeSpan.FromMinutes(2),
            new SearchSettings(Seed: 1, Threads: 1),
            MaxTrials: 40);

        var result = new TrialRunner(request, null).Run(split.TrainSet, roles);

        var learned = ml.MulticlassClassification.Evaluate(result.Model.Transform(split.TrainSet), "Species").MicroAccuracy;
        // The search's top score is the linear model's, which refit to 0.667 on these rows when nothing checked it: the guard ran
        // and another trainer's refit stands in.
        Assert.Equal("SdcaMaximumEntropy", result.TopScoring.Trainer);
        Assert.True(result.RefitNotReproduced, $"top {result.TopScoring.Trainer}, kept {result.Best.Trainer}");
        Assert.NotEqual("SdcaMaximumEntropy", result.Best.Trainer);
        Assert.NotNull(result.FallbackReason);
        Assert.True(learned >= 0.75, $"the model scores {learned} on the rows it was fitted on (validated {result.Best.Metrics!["micro_accuracy"]}); fallback: {result.FallbackReason}");
    }
}
