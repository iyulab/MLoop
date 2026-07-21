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

    [Theory]
    [InlineData(AutoTimePhase.ProbeStart, "probe")]
    [InlineData(AutoTimePhase.ProbeComplete, "main")]
    [InlineData(AutoTimePhase.ProbeConverged, "converged")]
    public void Phase_names_the_auto_time_transition_in_the_streams_own_vocabulary(AutoTimePhase phase, string expected)
    {
        var (emitter, sink) = Build();

        emitter.Phase(new TrainingProgress
        {
            TrialNumber = 4,
            TrainerName = "",
            MetricName = "",
            Metric = 0.7,
            ElapsedSeconds = 0,
            Phase = phase,
            ProbeTimeSeconds = 30,
            FinalTimeSeconds = 300
        });

        var e = Assert.Single(Events(sink));
        Assert.Equal("phase", e.GetProperty("event").GetString());
        Assert.Equal(expected, e.GetProperty("phase").GetString());
        Assert.Equal(30, e.GetProperty("probeTimeSec").GetInt32());
        Assert.Equal(300, e.GetProperty("timeLimitSec").GetInt32());
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
            BestTrainer = "LightGbmBinary",
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
