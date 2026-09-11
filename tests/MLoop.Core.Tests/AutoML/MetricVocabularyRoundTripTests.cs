using Microsoft.ML.AutoML;
using MLoop.Core.AutoML;
using MLoop.Core.Evaluation;

namespace MLoop.Core.Tests.AutoML;

/// <summary>
/// The optimizer reads a canonical name into an ML.NET enum (<c>*MetricFor</c>) and the trial
/// reporter writes the enum back as a canonical name (<c>Describe*Metric</c>). They are each
/// other's inverse and nothing used to say so — <c>precision</c> and <c>recall</c> were names the
/// reporter could write and the optimizer refused. This pins the round trip in both directions,
/// and that every name the optimizer accepts is one <see cref="MetricNames"/> knows.
/// </summary>
public class MetricVocabularyRoundTripTests
{
    [Fact]
    public void Every_name_the_binary_reporter_writes_is_one_the_binary_optimizer_accepts()
    {
        foreach (var metric in Enum.GetValues<BinaryClassificationMetric>())
        {
            var name = AutoMLRunner.DescribeBinaryMetric(metric).Name;
            var parsed = AutoMLRunner.BinaryMetricFor(name);
            Assert.True(parsed.HasValue, $"the reporter writes '{name}' but the optimizer does not accept it");
            Assert.Equal(name, AutoMLRunner.DescribeBinaryMetric(parsed.Value).Name);
            Assert.Equal(name, MetricNames.Canonical(name));
        }
    }

    [Fact]
    public void Every_name_the_multiclass_reporter_writes_is_one_the_multiclass_optimizer_accepts()
    {
        foreach (var metric in Enum.GetValues<MulticlassClassificationMetric>())
        {
            var name = AutoMLRunner.DescribeMulticlassMetric(metric).Name;
            var parsed = AutoMLRunner.MulticlassMetricFor(name);
            Assert.True(parsed.HasValue, $"the reporter writes '{name}' but the optimizer does not accept it");
            Assert.Equal(name, AutoMLRunner.DescribeMulticlassMetric(parsed.Value).Name);
            Assert.Equal(name, MetricNames.Canonical(name));
        }
    }

    [Fact]
    public void Every_name_the_regression_reporter_writes_is_one_the_regression_optimizer_accepts()
    {
        foreach (var metric in Enum.GetValues<RegressionMetric>())
        {
            var name = AutoMLRunner.DescribeRegressionMetric(metric).Name;
            var parsed = AutoMLRunner.RegressionMetricFor(name);
            Assert.True(parsed.HasValue, $"the reporter writes '{name}' but the optimizer does not accept it");
            Assert.Equal(name, AutoMLRunner.DescribeRegressionMetric(parsed.Value).Name);
            Assert.Equal(name, MetricNames.Canonical(name));
        }
    }

    /// <summary>
    /// The other direction: a canonical name the vocabulary lists for one of the three AutoML
    /// families is accepted by that family's optimizer — a name added to <see cref="MetricNames"/>
    /// without a switch arm would otherwise be "known" to validate and unknown to training.
    /// </summary>
    [Fact]
    public void Every_canonical_name_is_accepted_by_at_least_one_optimizer_or_is_a_task_primary()
    {
        foreach (var name in MetricNames.All)
        {
            var accepted = AutoMLRunner.BinaryMetricFor(name).HasValue
                        || AutoMLRunner.MulticlassMetricFor(name).HasValue
                        || AutoMLRunner.RegressionMetricFor(name).HasValue
                        || TaskMetadata.AllPrimaryMetrics.Contains(name)
                        || name == "davies_bouldin_index"; // clustering's second metric: reported, never optimized
            Assert.True(accepted, $"'{name}' is in the vocabulary but no optimizer accepts it");
        }
    }

    /// <summary>The spelling that shipped in the examples for months, read the way it was meant.</summary>
    [Fact]
    public void The_examples_F1Score_reaches_the_optimizer_as_F1()
    {
        var canonical = MetricNames.Canonical("binary-classification", "F1Score");
        Assert.Equal(BinaryClassificationMetric.F1Score, AutoMLRunner.BinaryMetricFor(canonical!));
    }
}
