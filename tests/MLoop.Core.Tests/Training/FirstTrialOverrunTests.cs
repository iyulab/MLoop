using MLoop.Core.Training;
using MLoop.Search;

namespace MLoop.Core.Tests.Training;

/// <summary>A search always finishes its first trial; when that one trial is longer than the whole budget the run says so.</summary>
public class FirstTrialOverrunTests
{
    private static TrialResult Trial(string trainer, double? seconds) => new() { Trainer = trainer, RuntimeSeconds = seconds };

    [Fact]
    public void AFirstTrialLongerThanTheBudget_IsNamedWithItsTrainerAndSeconds()
    {
        var sentence = TrainingRunner.FirstTrialOverrun([Trial("FastTreeOva", 62.5), Trial("SdcaMaximumEntropy", 2)], 30);

        Assert.NotNull(sentence);
        Assert.Contains("FastTreeOva", sentence, StringComparison.Ordinal);
        Assert.Contains("took 63 s", sentence, StringComparison.Ordinal);
        Assert.Contains("30 s time budget", sentence, StringComparison.Ordinal);
        Assert.Contains("--time", sentence, StringComparison.Ordinal);
    }

    [Fact]
    public void AFirstTrialThatFitsTheBudget_SaysNothing()
        => Assert.Null(TrainingRunner.FirstTrialOverrun([Trial("FastTreeOva", 5.8), Trial("FastTreeOva", 80)], 30));

    [Fact]
    public void ALaterTrialThatRunsOver_IsNotTheFirstTrialsDoing()
        => Assert.Null(TrainingRunner.FirstTrialOverrun([Trial("A", 1), Trial("B", 90)], 30));

    [Fact]
    public void TrialsWithNoRecordedTime_SayNothing()
    {
        Assert.Null(TrainingRunner.FirstTrialOverrun([], 30));
        Assert.Null(TrainingRunner.FirstTrialOverrun([Trial("A", null)], 30));
    }
}
