using Microsoft.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Training;
using MLoop.Tests.Common;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// A ranking run compares its trainers on queries held out of the training partition and reports the score of the winner
/// fitted again on the whole partition, scored once on the test queries. Choosing on the test queries and reporting the same
/// score made the reported number better than the model by the margin of the choice.
/// </summary>
[Collection("FileSystem")]
[Trait(TestCategories.Category, TestCategories.Integration)]
public class RankingSelectionTests : IDisposable
{
    private readonly string _tmpDir = Path.Combine(Path.GetTempPath(), $"mloop_rankselect_{Guid.NewGuid():N}");

    public RankingSelectionTests() => Directory.CreateDirectory(_tmpDir);

    public void Dispose()
    {
        if (Directory.Exists(_tmpDir))
            Directory.Delete(_tmpDir, recursive: true);
    }

    private async Task<TrainedModel> TrainAsync(int queries, int documents)
    {
        var path = Path.Combine(_tmpDir, $"rank{queries}.csv");
        var random = new Random(queries);
        var lines = new List<string> { "query,f0,f1,label" };
        for (var q = 0; q < queries; q++)
        {
            for (var d = 0; d < documents; d++)
            {
                var f0 = random.NextDouble() * 2 - 1;
                var f1 = random.NextDouble() * 2 - 1;
                var label = Math.Clamp((int)Math.Round(2 + 1.5 * f0 + 0.7 * (random.NextDouble() - 0.5)), 0, 4);
                lines.Add(FormattableString.Invariant($"q{q},{f0:0.0000},{f1:0.0000},{label}"));
            }
        }
        await File.WriteAllLinesAsync(path, lines);

        var ctx = new MLContext(seed: 42);
        var runner = new TrainingRunner(ctx, new CsvDataLoader(ctx));
        return await runner.RunAsync(new TrainingConfig
        {
            ModelName = "rank",
            DataFile = path,
            LabelColumn = "label",
            GroupColumn = "query",
            Task = "ranking",
            TimeLimitSeconds = 30,
        }, null, CancellationToken.None);
    }

    [Fact]
    public async Task TheReportedScore_IsNotTheScoreTheWinnerWasChosenBy()
    {
        var result = await TrainAsync(queries: 60, documents: 10);

        Assert.NotEmpty(result.Trials);
        var winnerScore = result.Trials.Max(t => t.Metrics["ndcg"]);
        // The winner was scored on queries held out of its fitting; the report is the refitted model on the test queries.
        Assert.NotEqual(winnerScore, result.Metrics["ndcg"]);
        Assert.InRange(result.Metrics["ndcg"], 0.0, 1.0);
    }

    [Fact]
    public async Task TooFewQueriesToSpare_TakesTheFirstTrainerThatFits_AndReportsItsTestScore()
    {
        var result = await TrainAsync(queries: 9, documents: 20);

        // Nothing to choose with: one trial, and the only score there is is the test score.
        var trial = Assert.Single(result.Trials);
        Assert.Equal(trial.Metrics["ndcg"], result.Metrics["ndcg"]);
    }
}
