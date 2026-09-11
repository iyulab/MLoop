using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Models;
using MLoop.Core.Storage;

namespace MLoop.Tests.Infrastructure.FileSystem;

/// <summary>
/// Pins the Markdown page an experiment leaves beside its JSON. The page is a rendering of files
/// that already exist, so what these tests protect is that the rendering says what the files say —
/// in the reader's order, in one culture, and without a table breaking on the values in it.
/// </summary>
public class ExperimentReportTests
{
    private const string Root = "/proj";
    private const string Version = "0.0.0-test";

    private static TrialRecord Trial(int n, TrainerDescriptor trainer, string metric, double value) =>
        new()
        {
            TrialNumber = n,
            Trainer = trainer,
            Metrics = new Dictionary<string, double> { [metric] = value },
            RuntimeSeconds = n * 1.5
        };

    private static ExperimentData Completed(
        IReadOnlyList<TrialRecord>? trials = null,
        string? rankingMetric = "auc",
        InputSchemaInfo? schema = null,
        TrainerDescriptor? trainer = null,
        string dataFile = "/proj/datasets/train.csv") =>
        new()
        {
            ModelName = "churn",
            ExperimentId = "exp-007",
            Timestamp = new DateTime(2026, 9, 11, 3, 4, 5, DateTimeKind.Utc),
            Status = "completed",
            Task = "binary-classification",
            Config = new ExperimentConfig
            {
                DataFile = dataFile,
                LabelColumn = "Churn",
                TimeLimitSeconds = 90,
                Metric = "auc",
                TestSplit = 0.2,
                InputSchema = schema
            },
            Result = new ExperimentResult
            {
                BestTrainer = (trainer ?? TrainerDescriptor.Of("ReplaceMissingValues=>Concatenate=>LightGbmBinary")).Display,
                Trainer = trainer ?? TrainerDescriptor.Of("ReplaceMissingValues=>Concatenate=>LightGbmBinary"),
                TrainingTimeSeconds = 91.37
            },
            Metrics = new Dictionary<string, double> { ["auc"] = 0.91234, ["accuracy"] = 0.85 },
            Trials = trials ?? [],
            RankingMetric = rankingMetric
        };

    private static ExperimentData Failed() =>
        new()
        {
            ModelName = "default",
            ExperimentId = "exp-001",
            Timestamp = DateTime.UtcNow,
            Status = "failed",
            Task = "regression",
            Config = new ExperimentConfig
            {
                DataFile = "train.csv",
                LabelColumn = "price",
                TimeLimitSeconds = 15,
                Metric = "r_squared",
                TestSplit = 0.2
            },
            Result = new ExperimentResult { BestTrainer = "none", TrainingTimeSeconds = 15.2 }
        };

    [Fact]
    public void A_completed_experiment_reads_top_down_from_result_to_columns()
    {
        var schema = new InputSchemaInfo
        {
            CapturedAt = DateTime.UtcNow,
            Columns =
            [
                new ColumnSchema { Name = "Churn", DataType = "String", Purpose = "Label" },
                new ColumnSchema { Name = "Tenure", DataType = "Single", Purpose = "Feature" },
                new ColumnSchema { Name = "CustomerId", DataType = SchemaDataTypes.ExcludedIdentifier, Purpose = "Exclude" }
            ]
        };
        var trials = new[]
        {
            Trial(1, TrainerDescriptor.Of("Concatenate=>FastTreeBinary"), "auc", 0.88),
            Trial(2, TrainerDescriptor.Of("ReplaceMissingValues=>Concatenate=>LightGbmBinary"), "auc", 0.91234)
        };

        var page = ExperimentReport.Render(Completed(trials, schema: schema), [trials[1], trials[0]], Root, Version);

        Assert.StartsWith("# exp-007 — churn", page);
        // The Result section names the trainer alone and the pipeline as its own row.
        Assert.Contains("| Best trainer | LightGbmBinary |", page);
        Assert.Contains("| Pipeline | ReplaceMissingValues=>Concatenate=>LightGbmBinary |", page);
        Assert.Contains("| Training time | 91.4 s |", page);
        // Metrics are written to four places, invariant culture, and sorted by name.
        Assert.Contains("| accuracy | 0.8500 |", page);
        Assert.Contains("| auc | 0.9123 |", page);
        Assert.True(page.IndexOf("| accuracy | 0.8500 |", StringComparison.Ordinal)
                    < page.IndexOf("| auc | 0.9123 |", StringComparison.Ordinal));
        // The leaderboard is in the ranked order it was given, best first — not completion order.
        Assert.Contains("best first by `auc` (higher is better)", page);
        Assert.Contains("| 1 | LightGbmBinary[^1] | 0.9123 | 3.0 s |", page);
        Assert.Contains("| 2 | FastTreeBinary[^2] | 0.8800 | 1.5 s |", page);
        Assert.Contains("[^1]: ReplaceMissingValues=>Concatenate=>LightGbmBinary", page);
        // The column table carries the exclusion reason the training warning gave.
        Assert.Contains("| CustomerId | - | Excluded (Identifier) |", page);
        Assert.Contains("| Tenure | Single | Feature |", page);
        // The data file is named from inside the project, not by the machine path it was given as.
        Assert.Contains("| Data | datasets/train.csv |", page.Replace('\\', '/'));
        Assert.DoesNotContain("/proj/", page.Replace('\\', '/'));
        // The footer points at the files it renders, by their layout names.
        Assert.Contains($"`{ExperimentLayout.MetadataFileName}`", page);
        Assert.Contains($"`{ExperimentLayout.LeaderboardFileName}`", page);
        Assert.Contains($"mloop {Version}", page);
    }

    /// <summary>
    /// The failure path saves the experiment from a catch block and then rethrows the training
    /// error. A renderer that threw on the failure record would replace that error with its own.
    /// </summary>
    [Fact]
    public void A_failed_experiment_renders_without_a_model_metrics_trials_or_schema()
    {
        var page = ExperimentReport.Render(Failed(), null, Root, Version);

        Assert.Contains("| Status | failed |", page);
        Assert.Contains("Training did not produce a model. It stopped after 15.2 s.", page);
        Assert.Contains("No metrics were recorded.", page);
        Assert.Contains("No trials were recorded.", page);
        Assert.Contains("No input schema was recorded.", page);
        Assert.DoesNotContain("| Best trainer |", page);
    }

    [Fact]
    public void A_pipe_in_a_value_does_not_open_a_new_column()
    {
        var schema = new InputSchemaInfo
        {
            CapturedAt = DateTime.UtcNow,
            Columns = [new ColumnSchema { Name = "a|b", DataType = "String", Purpose = "Feature" }]
        };
        var trainer = TrainerDescriptor.Of("Odd|Trainer");

        var page = ExperimentReport.Render(
            Completed([Trial(1, trainer, "auc", 0.5)], schema: schema, trainer: trainer), null, Root, Version);

        Assert.Contains(@"| a\|b | String | Feature |", page);
        Assert.Contains(@"| Best trainer | Odd\|Trainer |", page);
        Assert.Contains(@"| 1 | Odd\|Trainer | 0.5000 |", page);
    }

    /// <summary>
    /// When the ranking metric's direction is not known the store leaves the leaderboard unranked
    /// and says so; the page must say the same rather than presenting completion order as a ranking.
    /// </summary>
    [Fact]
    public void Trials_the_store_could_not_rank_are_listed_in_completion_order_and_say_so()
    {
        var trials = new[]
        {
            Trial(1, TrainerDescriptor.Of("A"), "custom_metric", 0.2),
            Trial(2, TrainerDescriptor.Of("B"), "custom_metric", 0.9)
        };

        var page = ExperimentReport.Render(Completed(trials, rankingMetric: "custom_metric"), null, Root, Version);

        Assert.Contains("in completion order — the direction of `custom_metric` is not known", page);
        Assert.DoesNotContain("best first", page);
        Assert.True(page.IndexOf("| 1 | A |", StringComparison.Ordinal) < page.IndexOf("| 2 | B |", StringComparison.Ordinal));
    }

    /// <summary>
    /// The configured metric is written in the user's spelling and the ranking metric in the
    /// canonical one, and nothing in the core says whether <c>F1Score</c> and <c>f1_score</c> are
    /// the same metric. The page states both and claims nothing about whether they differ — a
    /// sentence saying they did fired on an ordinary run (measured).
    /// </summary>
    [Fact]
    public void The_configured_and_ranking_metrics_are_both_named_without_a_claim_that_they_differ()
    {
        var trials = new[] { Trial(1, TrainerDescriptor.Of("A"), "f1_score", 0.7) };

        var page = ExperimentReport.Render(Completed(trials, rankingMetric: "f1_score"), trials, Root, Version);

        Assert.Contains("| Optimized for | auc |", page);
        Assert.Contains("best first by `f1_score`", page);
        Assert.DoesNotContain("instead", page);
    }

    [Fact]
    public void A_fallback_trainer_names_its_reason()
    {
        var trainer = new TrainerDescriptor
        {
            Name = "SdcaLogisticRegression",
            FallbackReason = "manual fallback: AutoML AUC failure"
        };

        var page = ExperimentReport.Render(Completed(trainer: trainer), null, Root, Version);

        Assert.Contains("| Best trainer | SdcaLogisticRegression |", page);
        Assert.Contains("| Fallback | manual fallback: AutoML AUC failure |", page);
        Assert.DoesNotContain("| Pipeline |", page);
    }

    [Fact]
    public void The_data_fingerprint_is_shown_when_recorded_and_absent_when_not()
    {
        var without = ExperimentReport.Render(Completed(), null, Root, Version);
        Assert.DoesNotContain("| Data hash |", without);

        var experiment = Completed();
        var withHash = new ExperimentData
        {
            ModelName = experiment.ModelName,
            ExperimentId = experiment.ExperimentId,
            Timestamp = experiment.Timestamp,
            Status = experiment.Status,
            Task = experiment.Task,
            Config = new ExperimentConfig
            {
                DataFile = experiment.Config.DataFile,
                DataFileHash = "sha256:" + new string('a', 64),
                LabelColumn = experiment.Config.LabelColumn,
                TimeLimitSeconds = experiment.Config.TimeLimitSeconds,
                Metric = experiment.Config.Metric,
                TestSplit = experiment.Config.TestSplit
            },
            Result = experiment.Result,
            Metrics = experiment.Metrics
        };

        var page = ExperimentReport.Render(withHash, null, Root, Version);
        Assert.Contains($"| Data hash | sha256:{new string('a', 64)} |", page);
    }

    [Fact]
    public void A_data_file_outside_the_project_is_named_as_given()
    {
        var elsewhere = Path.Combine(Path.GetTempPath(), "shared", "train.csv");

        var page = ExperimentReport.Render(Completed(dataFile: elsewhere), null, Root, Version);

        Assert.Contains($"| Data | {elsewhere} |", page);
    }
}
