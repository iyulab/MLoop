using MLoop.Ops.Interfaces;

namespace MLoop.Ops.Tests;

public class RollbackTargetTests
{
    private static PromotionRecord Record(string experiment, string? previous, string action = "promote") =>
        new("default", experiment, previous, action, null, DateTimeOffset.UnixEpoch);

    private static readonly PromotionRecord[] History =
    [
        Record("exp-001", null),
        Record("exp-002", "exp-001"),
        Record("exp-004", "exp-002"),
    ];

    [Fact]
    public void ReturnsWhatTheCurrentProductionReplaced()
        => Assert.Equal(("exp-002", null), RollbackTarget.Resolve(History, "exp-004"));

    [Fact]
    public void AFormerProductionCanBeNamed()
        => Assert.Equal(("exp-001", null), RollbackTarget.Resolve(History, "exp-004", "exp-001"));

    [Fact]
    public void AnExperimentThatNeverServedIsPromotesNotRollbacks()
    {
        var (target, error) = RollbackTarget.Resolve(History, "exp-004", "exp-003");

        Assert.Null(target);
        Assert.Contains("never been in production", error);
        Assert.Contains("mloop promote exp-003", error);
    }

    [Fact]
    public void TheFirstProductionHasNothingBeforeIt()
    {
        var (target, error) = RollbackTarget.Resolve(History, "exp-001");

        Assert.Null(target);
        Assert.Contains("--to", error);
    }

    [Fact]
    public void NoProductionMeansNothingToRollBack()
        => Assert.Null(RollbackTarget.Resolve(History, null).ExperimentId);

    [Fact]
    public void RollingBackTwiceUndoesTheRollback()
    {
        // The rollback is itself recorded, so "what production held before" is the model it replaced.
        var after = History.Append(Record("exp-002", "exp-004", "rollback")).ToArray();

        Assert.Equal(("exp-004", null), RollbackTarget.Resolve(after, "exp-002"));
    }
}
