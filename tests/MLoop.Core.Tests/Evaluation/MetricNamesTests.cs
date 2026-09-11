using MLoop.Core.Evaluation;

namespace MLoop.Core.Tests.Evaluation;

/// <summary>
/// Pins the one vocabulary a user's metric name is read through. Before it existed the optimizer,
/// <c>validate</c> and the shipped examples each had their own spellings, and a name the optimizer
/// did not know fell through to the task default without a word — <c>metric: F1Score</c> optimized
/// accuracy while every record said F1.
/// </summary>
public class MetricNamesTests
{
    [Theory]
    [InlineData("f1_score", "f1_score")]
    [InlineData("F1Score", "f1_score")]
    [InlineData("f1-score", "f1_score")]
    [InlineData("F1", "f1_score")]
    [InlineData("r_squared", "r_squared")]
    [InlineData("RSquared", "r_squared")]
    [InlineData("rSquared", "r_squared")]
    [InlineData("r2", "r_squared")]
    [InlineData("AUC", "auc")]
    [InlineData("AreaUnderRocCurve", "auc")]
    [InlineData("AreaUnderPrecisionRecallCurve", "auprc")]
    [InlineData("log-loss", "log_loss")]
    [InlineData("LogLoss", "log_loss")]
    [InlineData("MacroAccuracy", "macro_accuracy")]
    [InlineData("RootMeanSquaredError", "rmse")]
    [InlineData("MeanAbsoluteError", "mae")]
    [InlineData("PositivePrecision", "precision")]
    [InlineData("PositiveRecall", "recall")]
    [InlineData("  accuracy  ", "accuracy")]
    public void Canonical_reads_every_spelling_of_a_metric_as_one_name(string spelling, string expected)
    {
        Assert.Equal(expected, MetricNames.Canonical(spelling));
        Assert.True(MetricNames.IsKnown(spelling));
    }

    [Theory]
    [InlineData("nonexistent")]
    [InlineData("f2_score")]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData(null)]
    public void Canonical_returns_null_for_a_name_it_does_not_know_rather_than_a_default(string? spelling)
    {
        Assert.Null(MetricNames.Canonical(spelling));
        Assert.False(MetricNames.IsKnown(spelling));
    }

    [Fact]
    public void Auto_is_known_and_resolves_to_the_tasks_primary_metric()
    {
        Assert.Equal(MetricNames.Auto, MetricNames.Canonical("auto"));
        Assert.Equal(MetricNames.Auto, MetricNames.Canonical("AUTO"));
        Assert.Equal("accuracy", MetricNames.Canonical("binary-classification", "auto"));
        Assert.Equal("r_squared", MetricNames.Canonical("regression", "auto"));
        // A task with no primary keeps the auto value: nothing else would be true.
        Assert.Equal(MetricNames.Auto, MetricNames.Canonical("object-detection", "auto"));
    }

    /// <summary>
    /// The multiclass optimizer has always read <c>accuracy</c> as macro accuracy; the rule lives
    /// here now so the record names what was optimized.
    /// </summary>
    [Theory]
    [InlineData("multiclass-classification")]
    [InlineData("image-classification")]
    [InlineData("text-classification")]
    public void Accuracy_under_a_multiclass_task_means_macro_accuracy(string task)
    {
        Assert.Equal("macro_accuracy", MetricNames.Canonical(task, "accuracy"));
        Assert.Equal("macro_accuracy", MetricNames.Canonical(task, "Accuracy"));
    }

    [Fact]
    public void Accuracy_under_binary_classification_stays_accuracy()
    {
        Assert.Equal("accuracy", MetricNames.Canonical("binary-classification", "accuracy"));
    }

    /// <summary>Every task's primary metric is a name this vocabulary knows — the two authorities agree.</summary>
    [Fact]
    public void Every_task_primary_metric_is_a_known_name()
    {
        foreach (var primary in TaskMetadata.AllPrimaryMetrics)
            Assert.Equal(primary, MetricNames.Canonical(primary));
    }

    [Fact]
    public void Every_canonical_name_is_its_own_canonical_form()
    {
        foreach (var name in MetricNames.All)
            Assert.Equal(name, MetricNames.Canonical(name));
        Assert.Equal(MetricNames.All.Count, MetricNames.All.Distinct().Count());
    }
}
