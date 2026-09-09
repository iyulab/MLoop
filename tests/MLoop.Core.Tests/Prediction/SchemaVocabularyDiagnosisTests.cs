using Microsoft.ML;
using MLoop.Core.Models;
using MLoop.Core.Prediction;

namespace MLoop.Core.Tests.Prediction;

/// <summary>
/// A saved schema whose <c>DataType</c> is outside the <see cref="SchemaDataTypes"/> vocabulary
/// selects no feature column, and the run used to die a step later on a missing <c>Features</c>
/// column — naming the symptom and none of the columns responsible. These pin that the cause is
/// reported instead, and that a schema which is merely *featureless* is not mistaken for a broken one.
/// </summary>
public class SchemaVocabularyDiagnosisTests : IDisposable
{
    private readonly MLContext _ml = new(seed: 42);
    private readonly List<string> _tempFiles = new();

    public void Dispose()
    {
        foreach (var file in _tempFiles)
        {
            try { File.Delete(file); } catch { }
        }
    }

    private sealed class Point
    {
        public float X1 { get; set; }
        public float X2 { get; set; }
        public float Y { get; set; }
    }

    /// <summary>A real regression model that expects a single "Features" vector, saved to disk.</summary>
    private string SaveRegressionModel()
    {
        var data = _ml.Data.LoadFromEnumerable(Enumerable.Range(0, 20).Select(i => new Point
        {
            X1 = i,
            X2 = i * 0.5f,
            Y = 2 * i + 3 * (i * 0.5f) + 1,
        }));
        var featurized = _ml.Transforms.Concatenate("Features", "X1", "X2").Fit(data).Transform(data);
        var model = _ml.Regression.Trainers.Sdca("Y", featureColumnName: "Features").Fit(featurized);

        var path = Path.GetTempFileName();
        _tempFiles.Add(path);
        _ml.Model.Save(model, featurized.Schema, path);
        return path;
    }

    private static InputSchemaInfo SchemaWith(string featureDataType) => new()
    {
        CapturedAt = DateTime.UtcNow,
        Columns =
        [
            new ColumnSchema { Name = "X1", DataType = featureDataType, Purpose = "Feature" },
            new ColumnSchema { Name = "X2", DataType = featureDataType, Purpose = "Feature" },
            new ColumnSchema { Name = "Y", DataType = SchemaDataTypes.Numeric, Purpose = "Label" },
        ],
    };

    private static Dictionary<string, object>[] Rows() =>
    [
        new() { ["X1"] = 1.0, ["X2"] = 0.5 },
    ];

    [Fact]
    public void Predict_SchemaUsesDotNetTypeNames_NamesTheColumnsAndTheVocabulary()
    {
        // "Single" is the .NET type name, not the semantic one — the exact mistake a hand-written
        // config.json (or a test fixture) makes, and the one that produced "Could not find feature
        // column 'Features'" with no mention of X1, X2, or the word "Single".
        var modelPath = SaveRegressionModel();
        var service = new PredictionService(_ml);

        var ex = Assert.Throws<InvalidOperationException>(
            () => service.Predict(Rows(), SchemaWith("Single"), modelPath, "regression", "Y"));

        Assert.Contains("X1", ex.Message);
        Assert.Contains("X2", ex.Message);
        Assert.Contains("Single", ex.Message);
        Assert.Contains(SchemaDataTypes.Numeric, ex.Message);
        // The old message is the one this exists to replace; it must not be what the caller sees.
        Assert.DoesNotContain("Could not find feature column", ex.Message);
    }

    [Fact]
    public void Predict_SchemaUsesTheVocabulary_DoesNotRaiseTheDiagnosis()
    {
        // The counter-case a scan-style guard needs: a correct schema must go through, or the check
        // above would pass for a reason that has nothing to do with the vocabulary.
        var modelPath = SaveRegressionModel();
        var service = new PredictionService(_ml);

        var result = service.Predict(Rows(), SchemaWith(SchemaDataTypes.Numeric), modelPath, "regression", "Y");

        Assert.Single(result.Rows);
        Assert.NotNull(result.Rows[0].Score);
    }

    [Fact]
    public void IsKnown_AcceptsEveryNameTheVocabularyDefines_AndRejectsDotNetTypeNames()
    {
        Assert.All(SchemaDataTypes.All, name => Assert.True(SchemaDataTypes.IsKnown(name)));
        // Case-insensitive, the way every consumer compares these.
        Assert.True(SchemaDataTypes.IsKnown("numeric"));

        Assert.False(SchemaDataTypes.IsKnown("Single"));
        Assert.False(SchemaDataTypes.IsKnown("Double"));
        Assert.False(SchemaDataTypes.IsKnown("String"));
        Assert.False(SchemaDataTypes.IsKnown(null));
        Assert.False(SchemaDataTypes.IsKnown(""));
    }
}
