using MLoop.Core.Models;

namespace MLoop.Core.Tests.Models;

/// <summary>
/// The validated score of the model an experiment kept: the winner's mean over the validation folds and its standard error —
/// what the held-out table alone does not show.
/// </summary>
public class ValidatedScoreTests
{
    private static TrialRecord Trial(int number, string metric, double value, double? standardError, int folds = 5, bool selected = false, string? error = null) =>
        new()
        {
            TrialNumber = number,
            Trainer = TrainerDescriptor.Of("FastTree"),
            Metrics = error is null ? new Dictionary<string, double> { [metric] = value } : new Dictionary<string, double>(),
            RuntimeSeconds = 1,
            Validation = standardError is { } se
                ? new TrialValidation(folds, new Dictionary<string, double> { [metric] = se * Math.Sqrt(folds) }, new Dictionary<string, double> { [metric] = se })
                : null,
            Selected = selected,
            Error = error
        };

    [Fact]
    public void TheBestTrial_OnAHigherIsBetterMetric_IsTheWinner()
    {
        var score = ValidatedScore.OfWinner([Trial(1, "r_squared", 0.30, 0.10), Trial(2, "r_squared", 0.43, 0.08), Trial(3, "r_squared", 0.20, 0.05)], "r_squared");

        Assert.Equal(new ValidatedScore("r_squared", 0.43, 0.08, 5), score);
    }

    [Fact]
    public void TheBestTrial_OnALowerIsBetterMetric_IsTheLowestValue()
    {
        var score = ValidatedScore.OfWinner([Trial(1, "rmse", 12.0, 1.0), Trial(2, "rmse", 9.5, 0.7), Trial(3, "rmse", 15.0, 2.0)], "rmse");

        Assert.Equal(9.5, score!.Value);
        Assert.Equal(0.7, score.StandardError);
    }

    [Fact]
    public void TheSelectedTrial_IsTheWinner_EvenWhenAnotherScoredHigher()
    {
        // The search keeps the smallest model within one standard error of the top: the model on disk is the selected trial.
        var score = ValidatedScore.OfWinner([Trial(1, "auc", 0.95, 0.01), Trial(2, "auc", 0.94, 0.02, selected: true)], "auc");

        Assert.Equal(2, score is null ? 0 : (int)Math.Round(score.StandardError * 100));
        Assert.Equal(0.94, score!.Value);
    }

    [Fact]
    public void AFailedTrial_IsNeverTheWinner()
    {
        var score = ValidatedScore.OfWinner([Trial(1, "auc", 0, null, error: "boom"), Trial(2, "auc", 0.9, 0.03)], "auc");

        Assert.Equal(0.9, score!.Value);
    }

    [Fact]
    public void AWinnerWithoutASpread_HasNoValidatedScore()
    {
        // A path that scores one split records no standard error: nothing is said rather than a spread made up.
        Assert.Null(ValidatedScore.OfWinner([Trial(1, "auc", 0.9, null)], "auc"));
    }

    [Fact]
    public void NoTrialsOrNoMetric_HaveNoValidatedScore()
    {
        Assert.Null(ValidatedScore.OfWinner(null, "auc"));
        Assert.Null(ValidatedScore.OfWinner([], "auc"));
        Assert.Null(ValidatedScore.OfWinner([Trial(1, "auc", 0.9, 0.01)], null));
        Assert.Null(ValidatedScore.OfWinner([Trial(1, "auc", 0.9, 0.01)], "f1_score"));
    }
}
