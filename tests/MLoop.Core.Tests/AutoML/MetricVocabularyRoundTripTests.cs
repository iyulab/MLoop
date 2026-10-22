using MLoop.Core.Evaluation;

namespace MLoop.Core.Tests.AutoML;

/// <summary>
/// The names a search may optimize are <see cref="MetricNames.OptimizableFor"/>, read by the trainer search
/// as they are — there is no second list. These pin that the vocabulary and that list agree: every name the
/// list gives is canonical, and every canonical name is optimizable somewhere or is listed here as one that
/// is reported and never optimized. (The list used to have a copy — a switch into ML.NET AutoML's enums —
/// and <c>precision</c> and <c>recall</c> were names the records could carry and the optimizer refused.)
/// </summary>
public class MetricVocabularyRoundTripTests
{
    [Theory]
    [InlineData("binary-classification")]
    [InlineData("multiclass-classification")]
    [InlineData("regression")]
    public void Every_optimizable_name_is_canonical(string task)
    {
        foreach (var name in MetricNames.OptimizableFor(task)!)
            Assert.Equal(name, MetricNames.Canonical(name));
    }

    [Fact]
    public void Every_canonical_name_is_optimizable_for_a_task_or_is_only_reported()
    {
        string[] reportedOnly =
        [
            "davies_bouldin_index",  // clustering's second metric
            "mape",                  // forecasting's second metric
            "exact_match",           // question answering's second metric
            "log_loss_reduction",    // refused as an optimizing metric (MetricNames.Rejection)
            CalibrationError.MetricName,
        ];
        var optimizable = new[] { "binary-classification", "multiclass-classification", "regression" }
            .SelectMany(task => MetricNames.OptimizableFor(task)!)
            .ToHashSet();

        foreach (var name in MetricNames.All)
            Assert.True(optimizable.Contains(name) || TaskMetadata.AllPrimaryMetrics.Contains(name) || reportedOnly.Contains(name),
                $"'{name}' is in the vocabulary but no search optimizes it");
    }

    /// <summary>The spelling that shipped in the examples for months, read the way it was meant.</summary>
    [Fact]
    public void The_examples_F1Score_reaches_the_optimizer_as_F1()
    {
        var canonical = MetricNames.Canonical("binary-classification", "F1Score");
        Assert.Equal("f1_score", canonical);
        Assert.Contains(canonical, MetricNames.OptimizableFor("binary-classification")!);
    }
}
