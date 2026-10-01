using Microsoft.ML;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.ML;

/// <summary>
/// The prediction CSV names a multiclass <c>Score</c> slot after its class, as <c>--json</c> does.
/// </summary>
public class PredictionCsvHeaderTests : IDisposable
{
    private readonly string _file = Path.Combine(Path.GetTempPath(), $"mloop-header-{Guid.NewGuid():N}.csv");

    public void Dispose()
    {
        if (File.Exists(_file)) File.Delete(_file);
    }

    private sealed class Row
    {
        public float X { get; set; }
        public string Label { get; set; } = "";
    }

    private static DataViewSchema ScoredSchema(params string[] classes)
    {
        var ml = new MLContext(seed: 1);
        var rows = classes.SelectMany((c, i) => new[] { new Row { X = i, Label = c }, new Row { X = i + 0.1f, Label = c } });
        var data = ml.Data.LoadFromEnumerable(rows);
        var model = ml.Transforms.Conversion.MapValueToKey("Label")
            .Append(ml.Transforms.Concatenate("Features", "X"))
            .Append(ml.MulticlassClassification.Trainers.SingleThreadSdcaMaximumEntropy())
            .Fit(data);
        return model.Transform(data).Schema;
    }

    [Fact]
    public void A_multiclass_score_slot_is_named_after_its_class()
    {
        // The CSV used to say Score.1 = 0.97 while --json, for the same row, said dog = 0.97.
        var schema = ScoredSchema("cat", "dog", "fox");
        File.WriteAllLines(_file, ["placeholder", "row"]);

        PredictionEngine.FixVectorColumnHeaders(_file, schema);

        var header = File.ReadAllLines(_file)[0];
        Assert.Contains("Score.cat,Score.dog,Score.fox", header);
        Assert.DoesNotContain("Score.0", header);
    }

    [Fact]
    public void A_class_name_holding_a_comma_is_quoted_in_the_header()
    {
        var schema = ScoredSchema("a,b", "c");
        File.WriteAllLines(_file, ["placeholder", "row"]);

        PredictionEngine.FixVectorColumnHeaders(_file, schema);

        Assert.Contains("\"Score.a,b\",Score.c", File.ReadAllLines(_file)[0]);
    }
}
