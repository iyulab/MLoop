using FluentAssertions;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;

namespace MLoop.DataStore.Tests;

public class RetentionPrunerTests : IDisposable
{
    private static readonly DateTimeOffset Now = new(2026, 10, 22, 8, 0, 0, TimeSpan.Zero);
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"mloop-retention-{Guid.NewGuid():N}");

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
    }

    private string Write(string store, string model, string name)
    {
        var path = Path.Combine(_root, ".mloop", store, model, name);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, "{}\n");
        return path;
    }

    [Fact]
    public void Removes_TheDaysBeforeThePeriod_InBothStores_AndKeepsTodayAndThePeriod()
    {
        var old = Write("logs", "default", "2026-10-11.jsonl");
        var oldOther = Write("logs", "other", "2026-09-01.jsonl");
        var oldFeedback = Write("feedback", "default", "2026-10-11.jsonl");
        var firstKept = Write("logs", "default", "2026-10-12.jsonl");
        var today = Write("logs", "default", "2026-10-22.jsonl");
        var keptFeedback = Write("feedback", "default", "2026-10-20.jsonl");

        var report = RetentionPruner.Prune(_root, days: 10, Now);

        report.Cutoff.Should().Be(new DateOnly(2026, 10, 12));
        (report.LogFiles, report.FeedbackFiles).Should().Be((2, 1));
        report.Bytes.Should().Be(9);
        new[] { old, oldOther, oldFeedback }.Should().AllSatisfy(f => File.Exists(f).Should().BeFalse());
        new[] { firstKept, today, keptFeedback }.Should().AllSatisfy(f => File.Exists(f).Should().BeTrue());
    }

    [Fact]
    public void AFileWhoseNameIsNotADay_IsNeverTouched()
    {
        var notADay = Write("logs", "default", "archive.jsonl");
        var otherKind = Write("logs", "default", "2020-01-01.txt");

        RetentionPruner.Prune(_root, days: 1, Now).Files.Should().Be(0);

        File.Exists(notADay).Should().BeTrue();
        File.Exists(otherKind).Should().BeTrue();
    }

    [Fact]
    public void ADryRun_ReportsWithoutRemoving()
    {
        var old = Write("logs", "default", "2026-01-01.jsonl");

        RetentionPruner.Prune(_root, days: 30, Now, dryRun: true).LogFiles.Should().Be(1);

        File.Exists(old).Should().BeTrue();
    }

    [Fact]
    public void NoStores_IsNothingToDo_AndAPeriodBelowOneDay_IsRefused()
    {
        RetentionPruner.Prune(_root, days: 7, Now).Files.Should().Be(0);
        FluentActions.Invoking(() => RetentionPruner.Prune(_root, days: 0, Now)).Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public async Task TheFilesItRemoves_AreTheOnesTheStoresWrite()
    {
        // The pruner reads the stores' layout by name; this pins it against what they actually write.
        var logger = new FilePredictionLogger(_root);
        var feedback = new FileFeedbackCollector(_root);
        await logger.LogPredictionAsync("default", "exp-001", new Dictionary<string, object> { ["x"] = 1 }, "yes");
        var id = (await logger.GetLogsAsync("default")).Single().Id!;
        await feedback.RecordFeedbackAsync(id, "no");

        var report = RetentionPruner.Prune(_root, days: 1, DateTimeOffset.UtcNow.AddDays(5));

        (report.LogFiles, report.FeedbackFiles).Should().Be((1, 1));
        (await logger.GetLogsAsync("default")).Should().BeEmpty();
        (await feedback.GetFeedbackAsync("default")).Should().BeEmpty();
    }
}
