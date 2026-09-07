using MLoop.CLI.Commands;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Models;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// The reason given for a promotion is true of the run that just happened.
/// </summary>
/// <remarks>
/// A project's first training has no production model, so "Better accuracy than current production
/// model" named a comparison that was never made — and that sentence is among the first things a new
/// user is shown, one command into their first project. The presenter already received the
/// production model (or its absence) and did not look.
/// </remarks>
public class PromotionReasonTests
{
    [Fact]
    public void TheFirstModelIsNotDescribedAsBeatingSomething()
    {
        var output = Render(production: null, current: 0.85);

        Assert.DoesNotContain("Better", output, StringComparison.Ordinal);
        Assert.Contains("First model", output, StringComparison.Ordinal);
    }

    [Fact]
    public void AReplacementNamesWhatItBeatAndByHowMuch()
    {
        var output = Render(
            production: Production("exp-004", accuracy: 0.7100),
            current: 0.8500);

        Assert.Contains("exp-004", output, StringComparison.Ordinal);
        // Both numbers travel: a claim of "better" that does not show the two values cannot be checked.
        Assert.Contains("0.7100", output, StringComparison.Ordinal);
        Assert.Contains("0.8500", output, StringComparison.Ordinal);
    }

    [Fact]
    public void AReplacementWithNoRecordedMetricStillNamesWhatItReplaced()
    {
        var output = Render(production: Production("exp-002", accuracy: null), current: 0.85);

        Assert.Contains("exp-002", output, StringComparison.Ordinal);
        Assert.Contains("Better accuracy", output, StringComparison.Ordinal);
    }

    private static ModelInfo Production(string experimentId, double? accuracy) => new()
    {
        ModelName = "default",
        ExperimentId = experimentId,
        PromotedAt = DateTime.UnixEpoch,
        ModelPath = Path.Combine(Path.GetTempPath(), "unused"),
        Metrics = accuracy is null ? null : new Dictionary<string, double> { ["accuracy"] = accuracy.Value },
    };

    private static string Render(ModelInfo? production, double current)
    {
        var originalAnsiConsole = AnsiConsole.Console;
        var buffer = new StringWriter();
        try
        {
            AnsiConsole.Console = AnsiConsole.Create(new AnsiConsoleSettings
            {
                Out = new AnsiConsoleOutput(buffer)
            });

            TrainPresenter.DisplayPromotionResult(
                promoted: true,
                primaryMetric: "accuracy",
                result: new TrainingResult
                {
                    ExperimentId = "exp-005",
                    Trainer = TrainerDescriptor.Of("LightGbmBinary"),
                    Metrics = new Dictionary<string, double> { ["accuracy"] = current },
                    TrainingTimeSeconds = 1.0,
                    ModelPath = Path.Combine(Path.GetTempPath(), "unused", "model.zip"),
                },
                production: production,
                minThreshold: null);

            return buffer.ToString();
        }
        finally
        {
            AnsiConsole.Console = originalAnsiConsole;
        }
    }
}
