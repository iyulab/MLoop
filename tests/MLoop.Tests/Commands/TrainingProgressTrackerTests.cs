using MLoop.CLI.Commands;
using MLoop.Core.Models;

namespace MLoop.Tests.Commands;

/// <summary>
/// Guards the progress bar's arithmetic. This logic used to live inline in TrainCommand's callback
/// with no coverage at all, which is part of why the tabular tasks could report nothing for so long
/// without a test going red.
/// </summary>
public class TrainingProgressTrackerTests
{
    private static TrainingProgress Trial(double elapsedSeconds, string trainerName = "FastTreeBinary") => new()
    {
        TrialNumber = 1,
        TrainerName = trainerName,
        MetricName = "accuracy",
        Metric = 0.9,
        ElapsedSeconds = elapsedSeconds
    };

    private static TrainingProgress Phase(TrainingPhase phase, int probeSeconds = 0, int finalSeconds = 0) => new()
    {
        TrialNumber = 0,
        TrainerName = "",
        MetricName = "",
        Metric = 0,
        ElapsedSeconds = 0,
        Phase = phase,
        ProbeTimeSeconds = probeSeconds,
        FinalTimeSeconds = finalSeconds
    };

    [Fact]
    public void Percent_is_measured_against_the_configured_budget_when_time_is_fixed()
    {
        var tracker = new TrainingProgressTracker(configuredTimeLimitSeconds: 100);

        Assert.Equal(25, tracker.PercentFor(Trial(elapsedSeconds: 25)));
    }

    [Fact]
    public void Percent_switches_to_the_probe_budget_then_the_main_budget()
    {
        // The defect this pins: measuring both auto-time phases against the configured limit made
        // the bar crawl through the probe and then jump backwards when main training restarted the
        // trial clock at zero.
        var tracker = new TrainingProgressTracker(configuredTimeLimitSeconds: 300);

        tracker.EnterPhase(Phase(TrainingPhase.ProbeStart, probeSeconds: 20));
        Assert.Equal(20, tracker.BudgetSeconds);
        Assert.Equal(50, tracker.PercentFor(Trial(elapsedSeconds: 10)));

        tracker.EnterPhase(Phase(TrainingPhase.ProbeComplete, probeSeconds: 20, finalSeconds: 200));
        Assert.Equal(200, tracker.BudgetSeconds);
        Assert.Equal(5, tracker.PercentFor(Trial(elapsedSeconds: 10)));
    }

    [Fact]
    public void Percent_stops_just_short_of_complete_so_the_bar_finishes_only_when_training_does()
    {
        var tracker = new TrainingProgressTracker(configuredTimeLimitSeconds: 10);

        Assert.Equal(99, tracker.PercentFor(Trial(elapsedSeconds: 999)));
    }

    [Fact]
    public void Percent_is_unavailable_rather_than_invented_when_no_budget_is_known()
    {
        var tracker = new TrainingProgressTracker(configuredTimeLimitSeconds: 0);

        Assert.Null(tracker.PercentFor(Trial(elapsedSeconds: 5)));
    }

    [Fact]
    public void Converged_phase_leaves_the_budget_alone()
    {
        var tracker = new TrainingProgressTracker(configuredTimeLimitSeconds: 300);
        tracker.EnterPhase(Phase(TrainingPhase.ProbeStart, probeSeconds: 20));

        tracker.EnterPhase(Phase(TrainingPhase.ProbeConverged, probeSeconds: 20));

        Assert.Equal(20, tracker.BudgetSeconds);
    }

    [Fact]
    public void MainStart_measures_against_the_announced_fixed_budget()
    {
        // A fixed-budget run announces its window with MainStart; the tracker adopts that budget
        // the same way it adopts auto-time's main budget, so both modes share one arithmetic.
        var tracker = new TrainingProgressTracker(configuredTimeLimitSeconds: 300);

        tracker.EnterPhase(Phase(TrainingPhase.MainStart, finalSeconds: 60));

        Assert.Equal(60, tracker.BudgetSeconds);
        Assert.Equal(50, tracker.PercentFor(Trial(elapsedSeconds: 30)));
    }

    [Fact]
    public void Complete_phase_leaves_the_budget_alone()
    {
        // Complete carries no budget (FinalTimeSeconds is 0 on it); switching to it would divide
        // by zero-or-nothing. The window is over — whatever budget was in force stays.
        var tracker = new TrainingProgressTracker(configuredTimeLimitSeconds: 300);
        tracker.EnterPhase(Phase(TrainingPhase.MainStart, finalSeconds: 60));

        tracker.EnterPhase(Phase(TrainingPhase.Complete));

        Assert.Equal(60, tracker.BudgetSeconds);
    }

    [Theory]
    [InlineData("ReplaceMissingValues=>Concatenate=>FastTreeBinary", "FastTreeBinary")]
    [InlineData("LightGbmBinary", "LightGbmBinary")]
    [InlineData("A=> SdcaLogisticRegressionBinary ", "SdcaLogisticRegressionBinary")]
    [InlineData("", "(unknown)")]
    [InlineData(null, "(unknown)")]
    public void Trainer_name_is_reduced_to_the_trainer(string? full, string expected)
    {
        Assert.Equal(expected, TrainingProgressTracker.ShortTrainerName(full));
    }

    [Fact]
    public void A_fallback_trainer_name_renders_as_text_not_markup()
    {
        // The manual fallback names itself with a bracketed note; unescaped, Spectre read
        // "[manual fallback: …]" as a style and threw on the progress refresh thread.
        var p = new MLoop.Core.Models.TrainingProgress
        {
            TrialNumber = 1,
            TrainerName = "SdcaLogisticRegression [manual fallback: AutoML AUC failure]",
            Metric = 0.8,
            MetricName = "accuracy",
            ElapsedSeconds = 1
        };

        var description = TrainingProgressTracker.TrialDescription(p);
        _ = new Spectre.Console.Markup(description); // parses — this constructor is where it threw
        var text = Spectre.Console.Markup.Remove(description);

        Assert.Contains("[manual fallback: AutoML AUC failure]", text);
        Assert.Contains("accuracy=0.8000", text);
    }

    [Theory]
    [InlineData("my[model]")]
    [InlineData("plain")]
    public void A_model_name_renders_as_text_in_start_and_finalize_descriptions(string name)
    {
        var start = new Spectre.Console.Markup(TrainingProgressTracker.StartDescription(name));
        var finalize = TrainingProgressTracker.PhaseDescription(
            new MLoop.Core.Models.TrainingProgress
            {
                TrialNumber = 0, TrainerName = "", Metric = 0, MetricName = "", ElapsedSeconds = 0,
                Phase = MLoop.Core.Models.TrainingPhase.Complete
            }, name);

        Assert.NotNull(start);
        Assert.NotNull(new Spectre.Console.Markup(finalize!));
    }

    private static TrainingProgress Epoch(int epoch, int max, double elapsed) => new()
    {
        TrialNumber = 0, TrainerName = "NER (NAS-BERT)", Metric = 0, MetricName = "", ElapsedSeconds = elapsed,
        Phase = TrainingPhase.Epoch, Epoch = epoch, MaxEpochs = max
    };

    [Fact]
    public void An_epoch_moves_the_bar_by_epochs_done_not_time_spent()
    {
        Assert.Equal(30, TrainingProgressTracker.FitPercent(Epoch(3, 10, elapsed: 5000)));
        // The last epoch is not the end: evaluation and saving follow.
        Assert.Equal(99, TrainingProgressTracker.FitPercent(Epoch(10, 10, elapsed: 100)));
        Assert.Null(TrainingProgressTracker.FitPercent(Trial(10)));
    }

    [Fact]
    public void An_epoch_says_how_far_along_and_about_how_long_is_left()
    {
        // Three epochs in four minutes: seven more take about nine minutes and a third.
        var text = TrainingProgressTracker.PhaseDescription(Epoch(3, 10, elapsed: 240), "default")!;

        Assert.Contains("Epoch 3/10", text);
        Assert.Contains("about 9m left", text);
        Assert.NotNull(new Spectre.Console.Markup(text));
    }

    private static TrainingProgress Featurized(int step, int steps, double elapsed) => new()
    {
        TrialNumber = 0, TrainerName = "Image classification (TensorFlow)", Metric = 0, MetricName = "",
        ElapsedSeconds = elapsed, Phase = TrainingPhase.Featurize, Step = step, Steps = steps
    };

    [Fact]
    public void Featurization_moves_the_bar_and_estimates_what_is_left()
    {
        // 100 of 400 images in two minutes: the other 300 take about six.
        var p = Featurized(100, 400, elapsed: 120);

        Assert.Equal(25, TrainingProgressTracker.FitPercent(p));
        var text = TrainingProgressTracker.PhaseDescription(p, "default")!;
        Assert.Contains("Featurizing 100/400", text);
        Assert.Contains("about 6m left", text);
        Assert.NotNull(new Spectre.Console.Markup(text));
    }

    [Fact]
    public void An_early_stopping_epoch_names_its_cap_as_a_cap_and_predicts_no_end()
    {
        // 200 is a cap the fit usually stops far short of; dividing by it would promise hours.
        var p = new TrainingProgress
        {
            TrialNumber = 0, TrainerName = "Image classification (TensorFlow)", Metric = 0, MetricName = "",
            ElapsedSeconds = 60, Phase = TrainingPhase.Epoch, Epoch = 12, MaxEpochs = 200, StopsEarly = true
        };

        var text = TrainingProgressTracker.PhaseDescription(p, "default")!;
        Assert.Contains("Epoch 12", text);
        Assert.Contains("at most 200", text);
        Assert.DoesNotContain("12/200", text);
        Assert.DoesNotContain("left", text);
        Assert.NotNull(new Spectre.Console.Markup(text));
    }

    [Fact]
    public void The_last_epoch_promises_no_remaining_time()
    {
        var text = TrainingProgressTracker.PhaseDescription(Epoch(10, 10, elapsed: 800), "default")!;

        Assert.Contains("Epoch 10/10", text);
        Assert.DoesNotContain("left", text);
    }

    [Theory]
    [InlineData(0.2, "1s")]
    [InlineData(40, "40s")]
    [InlineData(720, "12m")]
    [InlineData(3900, "1h 05m")]
    public void A_duration_reads_as_a_person_would_say_it(double seconds, string expected) =>
        Assert.Equal(expected, TrainingProgressTracker.FormatDuration(TimeSpan.FromSeconds(seconds)));
}
