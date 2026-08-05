using System.Text.Json;
using MLoop.CLI.Commands;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Models;

namespace MLoop.Tests.Commands;

/// <summary>
/// Pins the <c>train --json</c> event stream. A consumer running mloop as a subprocess parses this
/// line by line, so the shapes here are a contract: one JSON object per line, nothing else on stdout.
/// </summary>
public class TrainJsonEmitterTests
{
    private static (TrainJsonEmitter Emitter, StringWriter Sink) Build()
    {
        var sink = new StringWriter();
        return (new TrainJsonEmitter(sink), sink);
    }

    private static List<JsonElement> Events(StringWriter sink) =>
        [.. sink.ToString()
            .Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries)
            .Select(line => JsonDocument.Parse(line).RootElement)];

    [Fact]
    public void Trial_carries_the_number_trainer_metric_and_elapsed_time()
    {
        var (emitter, sink) = Build();

        emitter.Trial(new TrainingProgress
        {
            TrialNumber = 7,
            TrainerName = "Concatenate=>FastTreeBinary",
            MetricName = "accuracy",
            Metric = 0.8123,
            ElapsedSeconds = 4.21
        });

        var e = Assert.Single(Events(sink));
        Assert.Equal("trial", e.GetProperty("event").GetString());
        Assert.Equal(7, e.GetProperty("n").GetInt32());
        Assert.Equal("Concatenate=>FastTreeBinary", e.GetProperty("trainer").GetString());
        Assert.Equal("accuracy", e.GetProperty("metric").GetString());
        Assert.Equal(0.8123, e.GetProperty("value").GetDouble());
        Assert.Equal(4210, e.GetProperty("elapsedMs").GetInt64());
        Assert.False(string.IsNullOrWhiteSpace(e.GetProperty("ts").GetString()));
    }

    /// <summary>
    /// Each boundary carries exactly the facts it has — a field the phase cannot know (a probe
    /// time on a fixed-budget run, a metric before training starts) is omitted, never zero-filled.
    /// The name column doubles as the shared-vocabulary pin: a fixed budget's MainStart and
    /// auto-time's ProbeComplete both say "main", so consumers never branch on budgeting mode.
    /// </summary>
    [Theory]
    [InlineData(TrainingPhase.ProbeStart, "probe", true, false, false, false, false)]
    [InlineData(TrainingPhase.ProbeComplete, "main", true, true, true, true, false)]
    [InlineData(TrainingPhase.ProbeConverged, "converged", true, false, true, true, false)]
    [InlineData(TrainingPhase.MainStart, "main", false, true, false, false, false)]
    [InlineData(TrainingPhase.Complete, "complete", false, false, true, false, true)]
    public void Phase_names_the_boundary_and_carries_only_the_facts_it_has(
        TrainingPhase phase, string expected,
        bool probe, bool budget, bool trials, bool metric, bool elapsed)
    {
        var (emitter, sink) = Build();

        emitter.Phase(new TrainingProgress
        {
            TrialNumber = 4,
            TrainerName = "",
            MetricName = "",
            Metric = 0.7,
            ElapsedSeconds = 12.5,
            Phase = phase,
            ProbeTimeSeconds = 30,
            FinalTimeSeconds = 300
        });

        var e = Assert.Single(Events(sink));
        Assert.Equal("phase", e.GetProperty("event").GetString());
        Assert.Equal(expected, e.GetProperty("phase").GetString());

        Assert.Equal(probe, e.TryGetProperty("probeTimeSec", out var p));
        if (probe) Assert.Equal(30, p.GetInt32());
        Assert.Equal(budget, e.TryGetProperty("timeLimitSec", out var b));
        if (budget) Assert.Equal(300, b.GetInt32());
        Assert.Equal(trials, e.TryGetProperty("trials", out var t));
        if (trials) Assert.Equal(4, t.GetInt32());
        Assert.Equal(metric, e.TryGetProperty("metric", out var m));
        if (metric) Assert.Equal(0.7, m.GetDouble());
        Assert.Equal(elapsed, e.TryGetProperty("elapsedMs", out var el));
        if (elapsed) Assert.Equal(12500, el.GetInt64());
    }

    [Fact]
    public void A_trial_update_is_not_a_phase_event()
    {
        // TrainCommand branches on Phase.HasValue; the emitter must not invent a phase for a trial.
        var (emitter, sink) = Build();

        emitter.Phase(new TrainingProgress
        {
            TrialNumber = 1,
            TrainerName = "X",
            MetricName = "auc",
            Metric = 0.9,
            ElapsedSeconds = 1
        });

        Assert.Empty(Events(sink));
    }

    [Fact]
    public void Result_is_one_line_carrying_the_experiment_and_its_metrics()
    {
        var (emitter, sink) = Build();

        emitter.Result(new TrainingResult
        {
            ExperimentId = "exp-003",
            Trainer = TrainerDescriptor.Of("LightGbmBinary"),
            Metrics = new Dictionary<string, double> { ["accuracy"] = 0.9012 },
            TrainingTimeSeconds = 298.5,
            ModelPath = "/models/default/staging/exp-003/model.zip"
        }, "default");

        var e = Assert.Single(Events(sink));
        Assert.Equal("result", e.GetProperty("event").GetString());
        Assert.Equal("default", e.GetProperty("model").GetString());
        Assert.Equal("exp-003", e.GetProperty("experimentId").GetString());
        Assert.Equal("LightGbmBinary", e.GetProperty("bestTrainer").GetString());
        Assert.Equal(0.9012, e.GetProperty("metrics").GetProperty("accuracy").GetDouble());
        Assert.Equal(298.5, e.GetProperty("trainingTimeSec").GetDouble());
    }

    /// <summary>
    /// The parts a consumer needs in order to use the trainer as an identifier. <c>bestTrainer</c>
    /// carries hyperparameters and fallback notes inside the name — that is what it has always
    /// rendered, and it keeps doing so — so the structured field is what makes parsing unnecessary.
    /// </summary>
    [Fact]
    public void Result_carries_the_trainer_in_parts_beside_its_display_form()
    {
        var (emitter, sink) = Build();

        emitter.Result(new TrainingResult
        {
            ExperimentId = "exp-004",
            Trainer = new TrainerDescriptor
            {
                Name = "SdcaLogisticRegression",
                FallbackReason = "manual fallback: AutoML AUC failure"
            },
            Metrics = new Dictionary<string, double> { ["accuracy"] = 0.96 },
            TrainingTimeSeconds = 21.3,
            ModelPath = "/models/default/staging/exp-004/model.zip"
        }, "default");

        var e = Assert.Single(Events(sink));
        Assert.Equal(
            "SdcaLogisticRegression [manual fallback: AutoML AUC failure]",
            e.GetProperty("bestTrainer").GetString());

        var trainer = e.GetProperty("trainer");
        Assert.Equal("SdcaLogisticRegression", trainer.GetProperty("name").GetString());
        Assert.Equal("manual fallback: AutoML AUC failure", trainer.GetProperty("fallbackReason").GetString());
        // No hyperparameters here, so the field is absent rather than an empty object claiming none.
        Assert.False(trainer.TryGetProperty("params", out _));
    }

    /// <summary>
    /// Hyperparameters are the other half of the same problem: <c>KMeans (k=3)</c> reads as a
    /// trainer name and is not one.
    /// </summary>
    [Fact]
    public void Result_names_hyperparameters_separately_from_the_trainer()
    {
        var (emitter, sink) = Build();

        emitter.Result(new TrainingResult
        {
            ExperimentId = "exp-005",
            Trainer = TrainerDescriptor.Of("SsaForecasting", ("window", 7), ("horizon", 3)),
            Metrics = new Dictionary<string, double> { ["mae"] = 1.2 },
            TrainingTimeSeconds = 4.0,
            ModelPath = "/models/default/staging/exp-005/model.zip"
        }, "default");

        var e = Assert.Single(Events(sink));
        Assert.Equal("SsaForecasting (window=7, horizon=3)", e.GetProperty("bestTrainer").GetString());

        var trainer = e.GetProperty("trainer");
        Assert.Equal("SsaForecasting", trainer.GetProperty("name").GetString());
        Assert.Equal("7", trainer.GetProperty("params").GetProperty("window").GetString());
        Assert.Equal("3", trainer.GetProperty("params").GetProperty("horizon").GetString());
        Assert.False(trainer.TryGetProperty("fallbackReason", out _));
    }

    [Fact]
    public void Every_event_is_its_own_line_so_the_stream_can_be_read_as_it_arrives()
    {
        var (emitter, sink) = Build();

        emitter.Warning("class imbalance");
        emitter.Trial(new TrainingProgress
        {
            TrialNumber = 1,
            TrainerName = "A",
            MetricName = "auc",
            Metric = 0.5,
            ElapsedSeconds = 1
        });
        emitter.Error("it broke");

        var events = Events(sink);
        Assert.Equal(["warning", "trial", "error"], events.Select(e => e.GetProperty("event").GetString()));
        Assert.Equal(3, sink.ToString().Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries).Length);
    }

    [Fact]
    public void Concurrent_reports_never_share_a_line()
    {
        // AutoML reports trials from its own worker threads. Two objects interleaved on one line
        // would make the whole stream unparseable, not just that record.
        var (emitter, sink) = Build();

        Parallel.For(0, 200, new ParallelOptions { MaxDegreeOfParallelism = 4 }, i =>
            emitter.Trial(new TrainingProgress
            {
                TrialNumber = i,
                TrainerName = $"T{i}",
                MetricName = "auc",
                Metric = i / 200.0,
                ElapsedSeconds = i
            }));

        var events = Events(sink);
        Assert.Equal(200, events.Count);
        Assert.All(events, e => Assert.Equal("trial", e.GetProperty("event").GetString()));
    }
}
