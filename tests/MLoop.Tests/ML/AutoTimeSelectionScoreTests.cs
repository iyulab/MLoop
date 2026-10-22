using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.AutoML;
using MLoop.Core.Models;

namespace MLoop.Tests.ML;

/// <summary>
/// Auto-time judges its probe — whether to stop early, and whether the probe's model beats the main run's — by the
/// search's validation score, never the test partition the run then reports. It once read the test score for both,
/// so the reported score had chosen its own model; and it read an error metric as if higher were better.
/// </summary>
public class AutoTimeSelectionScoreTests
{
    private static readonly TrainingConfig Config = new()
    {
        ModelName = "m",
        DataFile = "train.csv",
        LabelColumn = "y",
        Task = "regression",
        Metric = "rmse",
    };

    private static AutoMLResult Result(double? validation, string? ranking) => new()
    {
        Model = null!,
        Trainer = TrainerDescriptor.Of("FastTree"),
        Metrics = new Dictionary<string, double> { ["rmse"] = 99, ["r_squared"] = 0.5 },
        ValidationScore = validation,
        RankingMetric = ranking,
    };

    [Fact]
    public void ASearchedRun_IsJudgedByItsValidationScore_InItsMetricsDirection()
    {
        Assert.Equal((12.5, false), TrainingEngine.SelectionScore(Result(12.5, "rmse"), Config));
        Assert.Equal((0.93, true), TrainingEngine.SelectionScore(Result(0.93, "r_squared"), Config));
    }

    [Fact]
    public void ARunWithoutASearch_FallsBackToItsOnlyScore()
        => Assert.Equal((99, false), TrainingEngine.SelectionScore(Result(null, null), Config));
}
