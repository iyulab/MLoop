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

    [Fact]
    public void ASearchOnASample_SaysSoAndNamesTheRows()
    {
        var sentence = TrainingRunner.SampledSearchNotice(new SamplingOutcome(Rows: 800_000, SearchRows: 60_000, RefitRows: 800_000, ProbeSeconds: 2), 60);

        Assert.NotNull(sentence);
        Assert.Contains("stratified sample", sentence, StringComparison.Ordinal);
        Assert.Contains("fitted on all of them", sentence, StringComparison.Ordinal);
        Assert.Contains("--time", sentence, StringComparison.Ordinal);
    }

    [Fact]
    public void AModelFittedOnTheSample_SaysThatToo()
    {
        var sentence = TrainingRunner.SampledSearchNotice(new SamplingOutcome(1_000_000, 10_000, RefitRows: 10_000, ProbeSeconds: 9), 60);

        Assert.NotNull(sentence);
        Assert.Contains("the model was fitted on the same", sentence, StringComparison.Ordinal);
        Assert.DoesNotContain("fitted on all of them", sentence, StringComparison.Ordinal);
    }

    [Fact]
    public void ASearchOnAllTheRows_SaysNothingAboutASample()
    {
        Assert.Null(TrainingRunner.SampledSearchNotice(null, 60));
        Assert.Null(TrainingRunner.SampledSearchNotice(new SamplingOutcome(80_000, 80_000, 80_000, 0.4), 60));
    }
}
