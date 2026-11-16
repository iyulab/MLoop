using System.Text;
using MLoop.CLI.Commands;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>mloop predict --log</c> writes the prediction logs from the two CSVs it has — what was predicted from and what it answered —
/// and the interval a regression answer stated rides along (its bounds from the output file, its level from the model). Run end to
/// end over hand-made files, so what lands in the log line is pinned without training a model.
/// </summary>
public sealed class PredictLogIntervalTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"mloop-predict-log-{Guid.NewGuid():N}");

    public PredictLogIntervalTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try
        {
            Directory.Delete(_root, recursive: true);
        }
        catch (IOException)
        {
            // A temp directory a test leaves behind is not the test's result.
        }
    }

    private string Csv(string name, string text)
    {
        var path = Path.Combine(_root, name);
        File.WriteAllText(path, text, new UTF8Encoding(false));
        return path;
    }

    private async Task<IReadOnlyList<PredictionLogEntry>> LogAsync(string input, string output, double? level)
    {
        await PredictCommand.LogPredictionsAsync(_root, "m", "exp-001", Csv("in.csv", input), Csv("out.csv", output), level);
        return (await new FilePredictionLogger(_root).GetLogsAsync("m")).OrderBy(e => e.Input["x"].ToString(), StringComparer.Ordinal).ToList();
    }

    [Fact]
    public async Task ABandedRegressionLogsItsAnswerItsConfidenceAndTheIntervalItStated()
    {
        var logs = await LogAsync(
            "x\n1\n2\n",
            "x,Score,ScoreLowerBound,ScoreUpperBound,Confidence\n1,15.5,10.5,20.5,0.4\n2,30,25,35,0.5\n",
            level: 0.9);

        Assert.Equal(2, logs.Count);
        Assert.Equal("15.5", logs[0].Output.ToString());
        Assert.Equal(0.4, logs[0].Confidence);
        Assert.Equal(new PredictionInterval(10.5, 20.5, 0.9), logs[0].Interval);
        Assert.Equal(new PredictionInterval(25, 35, 0.9), logs[1].Interval);
    }

    [Fact]
    public async Task AnAnswerWithoutABandLogsNoInterval()
    {
        // A classification (or a plain regression): the line is what it was before intervals were kept.
        var logs = await LogAsync("x\n1\n", "x,PredictedLabel,Confidence\n1,yes,0.8\n", level: 0.9);

        var entry = Assert.Single(logs);
        Assert.Equal("yes", entry.Output.ToString());
        Assert.Null(entry.Interval);
    }

    [Fact]
    public async Task ABandWithoutALevelLogsNoIntervalButStillLogsThePrediction()
    {
        // The file carries the bounds, not the level: a model that states none leaves nothing to judge them by.
        var logs = await LogAsync("x\n1\n", "x,Score,ScoreLowerBound,ScoreUpperBound\n1,15.5,10.5,20.5\n", level: null);

        var entry = Assert.Single(logs);
        Assert.Equal("15.5", entry.Output.ToString());
        Assert.Null(entry.Interval);
    }

    [Fact]
    public async Task ABoundThatIsNotANumberDoesNotStopThePredictionBeingLogged()
    {
        // `NaN` cannot be written to a JSON log line; the row is logged without its interval, the others with theirs.
        var logs = await LogAsync(
            "x\n1\n2\n",
            "x,Score,ScoreLowerBound,ScoreUpperBound\n1,15.5,NaN,20.5\n2,30,25,35\n",
            level: 0.9);

        Assert.Equal(2, logs.Count);
        Assert.Null(logs[0].Interval);
        Assert.Equal(new PredictionInterval(25, 35, 0.9), logs[1].Interval);
    }
}
