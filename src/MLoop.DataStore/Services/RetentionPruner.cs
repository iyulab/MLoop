using System.Globalization;
using MLoop.Core.Storage;

namespace MLoop.DataStore.Services;

/// <summary>What one pass of <see cref="RetentionPruner"/> removed (or, in a dry run, would remove).</summary>
/// <param name="Cutoff">The first day kept: files of earlier days are removed.</param>
/// <param name="LogFiles">Prediction log day-files removed.</param>
/// <param name="FeedbackFiles">Feedback day-files removed.</param>
/// <param name="Bytes">Their combined size.</param>
public sealed record RetentionReport(DateOnly Cutoff, int LogFiles, int FeedbackFiles, long Bytes)
{
    public int Files => LogFiles + FeedbackFiles;
}

/// <summary>
/// Removes prediction logs and feedback older than a retention period. Both stores are written one
/// file per model per UTC day (<c>.mloop/logs/{model}/{yyyy-MM-dd}.jsonl</c>, the same under
/// <c>.mloop/feedback/</c>), so a day leaves whole: no file is rewritten, and a file whose name is not
/// a date is never touched.
/// </summary>
/// <remarks>
/// Logs and feedback share one period. Feedback is matched to a logged prediction, and a retraining
/// set joins the two, so feedback outliving its prediction's log could no longer be used — and a log
/// kept without its feedback is only the part that cannot be learned from.
/// </remarks>
public static class RetentionPruner
{
    /// <summary>
    /// Removes the day-files more than <paramref name="days"/> days before <paramref name="now"/>'s UTC
    /// day — today and the <paramref name="days"/> days before it are kept.
    /// </summary>
    public static RetentionReport Prune(string projectRoot, int days, DateTimeOffset now, bool dryRun = false)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(days, 1);
        var cutoff = DateOnly.FromDateTime(now.UtcDateTime).AddDays(-days);
        var mloop = Path.Combine(projectRoot, ".mloop");
        var (logFiles, logBytes) = PruneStore(Path.Combine(mloop, "logs"), cutoff, dryRun);
        var (feedbackFiles, feedbackBytes) = PruneStore(Path.Combine(mloop, "feedback"), cutoff, dryRun);
        return new RetentionReport(cutoff, logFiles, feedbackFiles, logBytes + feedbackBytes);
    }

    private static (int Files, long Bytes) PruneStore(string store, DateOnly cutoff, bool dryRun)
    {
        if (!Directory.Exists(store))
            return (0, 0);

        int files = 0;
        long bytes = 0;
        foreach (var modelDir in Directory.EnumerateDirectories(store))
        {
            foreach (var file in Directory.EnumerateFiles(modelDir, "*.jsonl"))
            {
                if (!TimestampKeys.TryParseDay(Path.GetFileNameWithoutExtension(file), out var day) || day >= cutoff)
                    continue;
                try
                {
                    var size = new FileInfo(file).Length;
                    if (!dryRun)
                        File.Delete(file);
                    files++;
                    bytes += size;
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    // Left for the next pass — a file being appended to today is never this old.
                }
            }
        }
        return (files, bytes);
    }
}
