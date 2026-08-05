using System.Text.Json;
using MLoop.Core.Models;
using Xunit;

namespace MLoop.Core.Tests.Models;

/// <summary>
/// The trainer used to be one hand-built string per task path; it is now parts plus one
/// composition. These pin the composition against the strings those paths actually produced, because
/// the whole point of splitting them was that nothing a user or a stored experiment sees changes —
/// <c>BestTrainer</c> is rendered in the CLI summary, in <c>list</c>, in <c>promote</c>, in the API,
/// and is written into <c>metadata.json</c>.
/// </summary>
public class TrainerDescriptorTests
{
    [Theory]
    // Plain names: AutoML pipelines, ranking/ts-anomaly winners, the DL handlers.
    [InlineData("ReplaceMissingValues=>Concatenate=>LightGbmRegression")]
    [InlineData("ImageClassification (TensorFlow)")]
    public void A_bare_name_renders_as_itself(string name)
        => Assert.Equal(name, TrainerDescriptor.Of(name).Display);

    [Fact]
    public void Hyperparameters_render_in_the_order_given()
    {
        Assert.Equal("KMeans (k=3)", TrainerDescriptor.Of("KMeans", ("k", 3)).Display);
        Assert.Equal(
            "SsaForecasting (window=7, horizon=3)",
            TrainerDescriptor.Of("SsaForecasting", ("window", 7), ("horizon", 3)).Display);
        Assert.Equal(
            "MatrixFactorization (rank=32, iter=20)",
            TrainerDescriptor.Of("MatrixFactorization", ("rank", 32), ("iter", 20)).Display);
        Assert.Equal("RandomizedPca (rank=20)", TrainerDescriptor.Of("RandomizedPca", ("rank", 20)).Display);
    }

    [Fact]
    public void A_fallback_reason_renders_in_brackets_after_the_name()
    {
        Assert.Equal(
            "SdcaLogisticRegression [manual fallback: AutoML AUC failure]",
            new TrainerDescriptor
            {
                Name = "SdcaLogisticRegression",
                FallbackReason = "manual fallback: AutoML AUC failure"
            }.Display);

        Assert.Equal(
            "FastTreeBinary [metric fallback: AreaUnderRocCurve→F1Score (AUC imbalance)]",
            new TrainerDescriptor
            {
                Name = "FastTreeBinary",
                FallbackReason = "metric fallback: AreaUnderRocCurve→F1Score (AUC imbalance)"
            }.Display);
    }

    /// <summary>
    /// An empty parameter list would render as <c>Name ()</c> and, more to the point, would claim
    /// the trainer was configured with nothing rather than that MLoop names nothing for it.
    /// </summary>
    [Fact]
    public void No_parameters_means_absent_not_empty()
    {
        Assert.Null(TrainerDescriptor.Of("KMeans").Parameters);
        Assert.Equal("KMeans", TrainerDescriptor.Of("KMeans").Display);
        Assert.Equal("KMeans", (TrainerDescriptor.Of("KMeans") with { Parameters = [] }).Display);
    }

    /// <summary>
    /// Formatted invariantly: a machine reading these must not see <c>0,5</c> where another locale
    /// writes <c>0.5</c>.
    /// </summary>
    [Fact]
    public void Numbers_are_formatted_invariantly()
        => Assert.Equal("T (rate=0.5)", TrainerDescriptor.Of("T", ("rate", 0.5)).Display);

    /// <summary>
    /// One shape on the wire, whichever artifact carries it — the descriptor goes into both the
    /// <c>result</c> event and <c>metadata.json</c>, and describing the same fact two ways is the
    /// problem this type exists to remove, not one to reintroduce one level down.
    /// </summary>
    [Fact]
    public void Parameters_are_written_as_one_object_and_read_back_in_order()
    {
        var trainer = TrainerDescriptor.Of("SsaForecasting", ("window", 7), ("horizon", 3));

        var json = JsonSerializer.Serialize(trainer);

        Assert.Contains("\"params\":{\"window\":\"7\",\"horizon\":\"3\"}", json);
        // The rendering is never stored: the artifacts that need it already carry it beside this.
        Assert.DoesNotContain("display", json, StringComparison.OrdinalIgnoreCase);

        var restored = JsonSerializer.Deserialize<TrainerDescriptor>(json)!;
        Assert.Equal(trainer.Display, restored.Display);
        Assert.Equal(["window", "horizon"], restored.Parameters!.Select(p => p.Key));
    }

    /// <summary>Absent parts stay absent — an explicit null claims MLoop looked and found nothing.</summary>
    [Fact]
    public void Absent_parts_are_omitted_from_the_wire_form()
    {
        var json = JsonSerializer.Serialize(TrainerDescriptor.Of("LightGbmBinary"));

        Assert.DoesNotContain("params", json);
        Assert.DoesNotContain("fallbackReason", json);
    }
}
