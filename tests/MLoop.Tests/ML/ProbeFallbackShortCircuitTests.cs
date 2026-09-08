using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.AutoML;
using MLoop.Core.Models;

namespace MLoop.Tests.ML;

/// <summary>
/// Under auto-time, the second AutoML pass exists to buy more trials. When the probe reports that
/// AutoML could not run on the data at all, there are no more trials to buy.
/// </summary>
/// <remarks>
/// Measured on a 50-row imbalanced binary set before this: the probe failed the AUC computation,
/// fell back to F1, failed again, and trained a direct SDCA pipeline — then main training ran and
/// took the identical path, so the user saw the same two fallback warnings twice and a
/// "Phase 2: Main training (130s)" line above a pass that spent ten seconds not searching. The
/// cause is a property of the data (a test split with one class absent), which a larger budget does
/// not change.
/// </remarks>
public class ProbeFallbackShortCircuitTests
{
    private static AutoMLResult ProbeResultWith(TrainerDescriptor trainer) => new()
    {
        Model = null!,
        Trainer = trainer,
        Metrics = new Dictionary<string, double> { ["accuracy"] = 0.8 },
    };

    [Fact]
    public void AutoMLUnavailableStopsTheSecondPass()
    {
        var probe = ProbeResultWith(new TrainerDescriptor
        {
            Name = "SdcaLogisticRegression",
            FallbackReason = "manual fallback: AutoML AUC failure",
            FallbackKind = TrainerFallbackKind.AutoMLUnavailable,
        });

        Assert.True(TrainingEngine.AnotherSearchWouldRepeatTheSameFailure(probe));
    }

    /// <summary>
    /// A substituted metric is not the same situation: AutoML ran, searched, and produced a model —
    /// it simply optimized something else. More time still buys more trials, so the second pass
    /// must still happen.
    /// </summary>
    [Fact]
    public void ASubstitutedMetricStillEarnsTheSecondPass()
    {
        var probe = ProbeResultWith(new TrainerDescriptor
        {
            Name = "LightGbmBinary",
            FallbackReason = "metric fallback: Accuracy→F1Score (AUC imbalance)",
            FallbackKind = TrainerFallbackKind.MetricSubstituted,
        });

        Assert.False(TrainingEngine.AnotherSearchWouldRepeatTheSameFailure(probe));
    }

    [Fact]
    public void AnOrdinaryProbeEarnsTheSecondPass()
    {
        Assert.False(TrainingEngine.AnotherSearchWouldRepeatTheSameFailure(
            ProbeResultWith(TrainerDescriptor.Of("LightGbmBinary"))));
    }

    [Fact]
    public void TheKindDefaultsToNoneAndDoesNotChangeTheRendering()
    {
        var plain = TrainerDescriptor.Of("LightGbmBinary");

        Assert.Equal(TrainerFallbackKind.None, plain.FallbackKind);
        Assert.Equal("LightGbmBinary", plain.Display);
    }
}
