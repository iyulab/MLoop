using System.Globalization;
using System.Text;
using MLoop.CLI.Infrastructure.Display;
using MLoop.Core.Evaluation;
using MLoop.Core.Models;
using MLoop.Core.Storage;

namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Renders an experiment as one Markdown page — the <c>report.md</c> that sits beside
/// <c>metadata.json</c>, <c>metrics.json</c> and <c>leaderboard.json</c> in the experiment directory.
/// </summary>
/// <remarks>
/// <para>
/// Everything on the page is already on disk in JSON. The report exists because those files are
/// written for programs: a person who wants to know what an experiment did — which trainer won,
/// what else was tried, which columns were dropped and why — has to open three files and know how
/// each is shaped. This page says it once, in the order a reader asks the questions, and renders in
/// any Markdown viewer including the repository host the project is pushed to.
/// </para>
/// <para>
/// The page is a <i>rendering</i>, never a source: no command reads it back, so it can be stale,
/// missing or hand-edited without anything downstream noticing. That is also why writing it must
/// never fail the experiment it describes — the store wraps the write and reports a warning.
/// </para>
/// <para>
/// Numbers are written in the invariant culture. The file is committed to the user's repository, and
/// a decimal separator that follows the generating machine's locale would make the same experiment
/// read differently on each contributor's checkout.
/// </para>
/// </remarks>
internal static class ExperimentReport
{
    private const string NoValue = "-";

    /// <summary>
    /// The page for <paramref name="experiment"/>.
    /// </summary>
    /// <param name="experiment">The experiment as it is being saved — complete or failed.</param>
    /// <param name="rankedTrials">
    /// The trials best-first, as the leaderboard orders them, or null when the ranking metric is one
    /// <see cref="MetricDirection"/> does not know — the trials are then listed in completion order
    /// and the page says so, rather than sorting by a guessed direction.
    /// </param>
    /// <param name="projectRoot">
    /// The project the experiment belongs to, so the data file is named relative to it — a machine
    /// path in a committed document tells every other reader where one contributor kept the file.
    /// </param>
    /// <param name="toolVersion">The mloop that wrote it, for the footer.</param>
    internal static string Render(
        ExperimentData experiment,
        IReadOnlyList<TrialRecord>? rankedTrials,
        string projectRoot,
        string toolVersion)
    {
        var sb = new StringBuilder();

        sb.AppendLine($"# {Escape(experiment.ExperimentId)} — {Escape(experiment.ModelName)}");
        sb.AppendLine();

        RenderOverview(sb, experiment, projectRoot);
        RenderResult(sb, experiment);
        RenderMetrics(sb, experiment);
        RenderTrials(sb, experiment, rankedTrials);
        RenderColumns(sb, experiment);

        sb.AppendLine("---");
        sb.AppendLine();
        sb.AppendLine($"Written by mloop {toolVersion} from `{ExperimentLayout.MetadataFileName}`, " +
                      $"`{ExperimentLayout.MetricsFileName}` and `{ExperimentLayout.LeaderboardFileName}` " +
                      "in this directory. Those files are the record; this page is a rendering of them " +
                      "and is not read by any command.");

        return sb.ToString();
    }

    private static void RenderOverview(StringBuilder sb, ExperimentData experiment, string projectRoot)
    {
        var config = experiment.Config;

        sb.AppendLine("| | |");
        sb.AppendLine("|---|---|");
        Row(sb, "Status", experiment.Status);
        Row(sb, "Task", experiment.Task);
        Row(sb, "Recorded at", TimestampDisplay.Local(experiment.Timestamp));
        Row(sb, "Data", InsideTheProject(config.DataFile, projectRoot));
        if (!string.IsNullOrEmpty(config.DataFileHash))
            Row(sb, "Data hash", config.DataFileHash);
        Row(sb, "Label", string.IsNullOrEmpty(config.LabelColumn) ? NoValue : config.LabelColumn);
        if (!string.IsNullOrEmpty(config.GroupColumn))
            Row(sb, "Group column", config.GroupColumn);
        if (!string.IsNullOrEmpty(config.UserColumn))
            Row(sb, "User column", config.UserColumn);
        if (!string.IsNullOrEmpty(config.ItemColumn))
            Row(sb, "Item column", config.ItemColumn);
        Row(sb, "Test split", Percent(config.TestSplit));
        var seconds = $"{config.TimeLimitSeconds.ToString(CultureInfo.InvariantCulture)} s";
        Row(sb, "Time limit", config.AutoTime == true ? $"auto — {seconds} granted" : seconds);
        Row(sb, "Optimized for", string.IsNullOrEmpty(config.Metric) ? NoValue : config.Metric);
        sb.AppendLine();
    }

    private static void RenderResult(StringBuilder sb, ExperimentData experiment)
    {
        sb.AppendLine("## Result");
        sb.AppendLine();

        // The recorded fact about a failed run is its status; the placeholder trainer name the
        // failure record carries is not compared here, so this page does not hold a second copy of it.
        var result = experiment.Result;
        var failed = string.Equals(experiment.Status, "failed", StringComparison.OrdinalIgnoreCase);
        if (result is null || failed)
        {
            sb.AppendLine(result is null
                ? "No result was recorded."
                : $"Training did not produce a model. It stopped after {Seconds(result.TrainingTimeSeconds)}.");
            sb.AppendLine();
            return;
        }

        sb.AppendLine("| | |");
        sb.AppendLine("|---|---|");
        Row(sb, "Best trainer", result.Trainer is null
            ? TrainerDisplay.Short(result.BestTrainer)
            : TrainerDisplay.Short(result.Trainer));
        var pipeline = result.Trainer?.Name ?? result.BestTrainer;
        if (TrainerDisplay.WasShortened(pipeline))
            Row(sb, "Pipeline", pipeline);
        if (!string.IsNullOrEmpty(result.Trainer?.FallbackReason))
            Row(sb, "Fallback", result.Trainer.FallbackReason);
        Row(sb, "Training time", Seconds(result.TrainingTimeSeconds));
        sb.AppendLine();
    }

    private static void RenderMetrics(StringBuilder sb, ExperimentData experiment)
    {
        sb.AppendLine("## Metrics");
        sb.AppendLine();

        if (experiment.Metrics is null || experiment.Metrics.Count == 0)
        {
            sb.AppendLine("No metrics were recorded.");
            sb.AppendLine();
            return;
        }

        sb.AppendLine("| Metric | Value |");
        sb.AppendLine("|---|---|");
        foreach (var (name, value) in experiment.Metrics.OrderBy(m => m.Key, StringComparer.Ordinal))
            Row(sb, name, Number(value));
        sb.AppendLine();
    }

    private static void RenderTrials(
        StringBuilder sb, ExperimentData experiment, IReadOnlyList<TrialRecord>? rankedTrials)
    {
        sb.AppendLine("## Trials");
        sb.AppendLine();

        if (experiment.Trials.Count == 0)
        {
            sb.AppendLine("No trials were recorded.");
            sb.AppendLine();
            return;
        }

        var metric = experiment.RankingMetric;
        var trials = rankedTrials ?? experiment.Trials;
        var count = experiment.Trials.Count.ToString(CultureInfo.InvariantCulture);

        if (rankedTrials is not null && metric is not null)
        {
            var direction = MetricDirection.IsLowerBetter(metric) ? "lower is better" : "higher is better";
            sb.AppendLine($"{count} trial(s), best first by `{metric}` ({direction}).");
        }
        else if (metric is not null)
        {
            sb.AppendLine($"{count} trial(s) in completion order — the direction of `{metric}` is not " +
                          "known, so they are not ranked.");
        }
        else
        {
            sb.AppendLine($"{count} trial(s) in completion order.");
        }

        // The configured metric is in the user's spelling (`F1Score`) and the ranking metric in the
        // canonical one (`f1_score`); nothing in the core says whether two such names are one metric,
        // so the page states both — "Optimized for" above, "ranked by" here — and does not claim
        // that they differ.
        sb.AppendLine();

        var metricHeader = metric ?? "Metric";
        sb.AppendLine($"| # | Trainer | {Escape(metricHeader)} | Runtime |");
        sb.AppendLine("|---|---|---|---|");

        var footnotes = new List<string>();
        var rank = 0;
        foreach (var trial in trials)
        {
            rank++;
            // The column shows the trainer alone; the pipeline in front of it and any fallback
            // reason go under the table as a footnote — the same split the terminal leaderboard makes.
            var shown = Escape(TrainerDisplay.Short(trial.Trainer));
            var footnote = TrainerDisplay.Footnote($"[^{rank}]:", trial.Trainer);
            if (footnote is not null)
            {
                shown += $"[^{rank}]";
                footnotes.Add(footnote);
            }

            var value = metric is not null && trial.Metrics.TryGetValue(metric, out var v)
                ? Number(v)
                : NoValue;

            sb.AppendLine($"| {rank.ToString(CultureInfo.InvariantCulture)} | {shown} | {value} | {Seconds(trial.RuntimeSeconds)} |");
        }
        sb.AppendLine();

        if (footnotes.Count > 0)
        {
            // A footnote is prose, not a cell: a pipe in it is harmless, a line break would end it.
            foreach (var footnote in footnotes)
                sb.AppendLine(OneLine(footnote));
            sb.AppendLine();
        }
    }

    private static void RenderColumns(StringBuilder sb, ExperimentData experiment)
    {
        sb.AppendLine("## Columns");
        sb.AppendLine();

        var schema = experiment.Config.InputSchema;
        if (schema is null || schema.Columns.Count == 0)
        {
            sb.AppendLine("No input schema was recorded.");
            sb.AppendLine();
            return;
        }

        sb.AppendLine("| Column | Type | Role |");
        sb.AppendLine("|---|---|---|");
        foreach (var column in schema.Columns)
        {
            // An excluded column carries the reason in its DataType slot (the same convention the
            // saved schema uses), so the row says what the warning at training time said.
            var role = string.Equals(column.Purpose, "Exclude", StringComparison.OrdinalIgnoreCase)
                ? $"Excluded ({column.DataType})"
                : column.Purpose;
            var type = string.Equals(column.Purpose, "Exclude", StringComparison.OrdinalIgnoreCase)
                ? NoValue
                : column.DataType;
            sb.AppendLine($"| {Escape(column.Name)} | {Escape(type)} | {Escape(role)} |");
        }
        sb.AppendLine();
    }

    private static void Row(StringBuilder sb, string label, string value) =>
        sb.AppendLine($"| {Escape(label)} | {Escape(value)} |");

    /// <summary>
    /// A cell value that cannot break the table it sits in: a pipe would open a new column and a
    /// line break a new row. Trainer pipelines carry <c>=&gt;</c>, hyperparameter lists carry commas
    /// and parentheses — none of those need escaping, only the two characters Markdown reads as
    /// table structure.
    /// </summary>
    private static string Escape(string? value) =>
        string.IsNullOrEmpty(value) ? NoValue : OneLine(value).Replace("|", "\\|");

    private static string OneLine(string value) =>
        value.Replace("\r\n", " ").Replace('\n', ' ').Replace('\r', ' ');

    private static string Number(double value) =>
        double.IsFinite(value) ? value.ToString("F4", CultureInfo.InvariantCulture) : NoValue;

    private static string Seconds(double seconds) =>
        $"{seconds.ToString("F1", CultureInfo.InvariantCulture)} s";

    private static string Percent(double fraction) =>
        $"{(fraction * 100).ToString("0.#", CultureInfo.InvariantCulture)}%";

    /// <summary>
    /// The data file as it reads from inside the project — the same choice the training summary
    /// makes for the model path. A file outside the project is named as it was given.
    /// </summary>
    private static string InsideTheProject(string dataFile, string projectRoot)
    {
        if (string.IsNullOrEmpty(dataFile) || !Path.IsPathRooted(dataFile) || string.IsNullOrEmpty(projectRoot))
            return string.IsNullOrEmpty(dataFile) ? NoValue : dataFile;

        var relative = Path.GetRelativePath(projectRoot, dataFile);
        return relative.StartsWith("..", StringComparison.Ordinal) || Path.IsPathRooted(relative)
            ? dataFile
            : relative;
    }
}
