using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Models;
using MLoop.Core.Prediction;

namespace MLoop.Core.Tests.Prediction;

public class PredictionServiceTests : IDisposable
{
    private readonly MLContext _mlContext = new(seed: 42);
    private readonly List<string> _tempFiles = new();

    public void Dispose()
    {
        foreach (var file in _tempFiles)
        {
            try { File.Delete(file); } catch { }
        }
    }

    private string SaveModel(ITransformer model, DataViewSchema schema)
    {
        var path = Path.GetTempFileName();
        _tempFiles.Add(path);
        _mlContext.Model.Save(model, schema, path);
        return path;
    }

    #region Regression

    [Fact]
    public void Predict_Regression_ReturnsScores()
    {
        // Train a tiny regression model
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new SimpleRegression { X = 1.0f, Y = 2.0f },
            new SimpleRegression { X = 2.0f, Y = 4.0f },
            new SimpleRegression { X = 3.0f, Y = 6.0f },
            new SimpleRegression { X = 4.0f, Y = 8.0f },
            new SimpleRegression { X = 5.0f, Y = 10.0f },
        });

        var pipeline = _mlContext.Transforms.Concatenate("Features", "X")
            .Append(_mlContext.Regression.Trainers.Sdca(labelColumnName: "Y"));
        var model = pipeline.Fit(data);
        var modelPath = SaveModel(model, data.Schema);

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Y", DataType = "Numeric", Purpose = "Label" }
            },
            CapturedAt = DateTime.UtcNow
        };

        var rows = new[]
        {
            new Dictionary<string, object> { ["X"] = 6.0f },
            new Dictionary<string, object> { ["X"] = 7.0f },
        };

        var service = new PredictionService(_mlContext);
        var result = service.Predict(rows, schema, modelPath, "regression", "Y");

        Assert.Equal("regression", result.TaskType);
        Assert.Equal(2, result.Rows.Count);
        Assert.NotNull(result.Rows[0].Score);
        Assert.NotNull(result.Rows[1].Score);
        // Rough linearity check: Y ≈ 2*X, so score for X=6 should be > score for X=5 area
        Assert.True(result.Rows[0].Score > 5.0, $"Expected score > 5.0, got {result.Rows[0].Score}");
    }

    [Fact]
    public void Predict_Regression_BooleanLabel_ConvertedToSingle()
    {
        // BUG-23: Boolean label (0/1) should be treated as Single for regression
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new SimpleRegression { X = 1.0f, Y = 0.0f },
            new SimpleRegression { X = 2.0f, Y = 0.0f },
            new SimpleRegression { X = 3.0f, Y = 1.0f },
            new SimpleRegression { X = 4.0f, Y = 1.0f },
            new SimpleRegression { X = 5.0f, Y = 1.0f },
        });

        var pipeline = _mlContext.Transforms.Concatenate("Features", "X")
            .Append(_mlContext.Regression.Trainers.Sdca(labelColumnName: "Y"));
        var model = pipeline.Fit(data);
        var modelPath = SaveModel(model, data.Schema);

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Y", DataType = "Boolean", Purpose = "Label" }
            },
            CapturedAt = DateTime.UtcNow
        };

        var rows = new[] { new Dictionary<string, object> { ["X"] = 3.5f } };

        var service = new PredictionService(_mlContext);
        var result = service.Predict(rows, schema, modelPath, "regression", "Y");

        Assert.Equal("regression", result.TaskType);
        Assert.Single(result.Rows);
        Assert.NotNull(result.Rows[0].Score);
    }

    #endregion

    #region Empty Input

    [Fact]
    public void Predict_EmptyRows_ReturnsEmptyResult()
    {
        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" },
            },
            CapturedAt = DateTime.UtcNow
        };

        var service = new PredictionService(_mlContext);
        var result = service.Predict(
            Array.Empty<Dictionary<string, object>>(),
            schema, "nonexistent.zip", "regression");

        Assert.Equal("regression", result.TaskType);
        Assert.Empty(result.Rows);
        Assert.NotNull(result.Warnings);
        Assert.Contains(result.Warnings, w => w.Contains("No input rows"));
    }

    [Fact]
    public void Predict_WithPreLoadedTransformer_ReturnsSameResultAsPathOverload()
    {
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new SimpleRegression { X = 1.0f, Y = 2.0f },
            new SimpleRegression { X = 2.0f, Y = 4.0f },
            new SimpleRegression { X = 3.0f, Y = 6.0f },
            new SimpleRegression { X = 4.0f, Y = 8.0f },
            new SimpleRegression { X = 5.0f, Y = 10.0f },
        });

        var pipeline = _mlContext.Transforms.Concatenate("Features", "X")
            .Append(_mlContext.Regression.Trainers.Sdca(labelColumnName: "Y"));
        var model = pipeline.Fit(data);
        var modelPath = SaveModel(model, data.Schema);

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Y", DataType = "Numeric", Purpose = "Label" }
            },
            CapturedAt = DateTime.UtcNow
        };

        var rows = new[]
        {
            new Dictionary<string, object> { ["X"] = 6.0f },
            new Dictionary<string, object> { ["X"] = 7.0f },
        };

        var service = new PredictionService(_mlContext);

        var pathResult = service.Predict(rows.Select(r => new Dictionary<string, object>(r)).ToArray(),
            schema, modelPath, "regression", "Y");

        var loadedModel = _mlContext.Model.Load(modelPath, out _);
        var transformerResult = service.Predict(rows.Select(r => new Dictionary<string, object>(r)).ToArray(),
            schema, loadedModel, "regression", "Y");

        Assert.Equal(pathResult.Rows.Count, transformerResult.Rows.Count);
        for (int i = 0; i < pathResult.Rows.Count; i++)
        {
            Assert.Equal(pathResult.Rows[i].Score, transformerResult.Rows[i].Score);
        }
    }

    [Fact]
    public void Predict_WithPreLoadedTransformer_ReusableAcrossCalls()
    {
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new SimpleRegression { X = 1.0f, Y = 2.0f },
            new SimpleRegression { X = 2.0f, Y = 4.0f },
            new SimpleRegression { X = 3.0f, Y = 6.0f },
        });

        var pipeline = _mlContext.Transforms.Concatenate("Features", "X")
            .Append(_mlContext.Regression.Trainers.Sdca(labelColumnName: "Y"));
        var model = pipeline.Fit(data);
        var modelPath = SaveModel(model, data.Schema);

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Y", DataType = "Numeric", Purpose = "Label" }
            },
            CapturedAt = DateTime.UtcNow
        };

        var service = new PredictionService(_mlContext);
        var loaded = _mlContext.Model.Load(modelPath, out _);

        var r1 = service.Predict(new[] { new Dictionary<string, object> { ["X"] = 4.0f } },
            schema, loaded, "regression", "Y");
        var r2 = service.Predict(new[] { new Dictionary<string, object> { ["X"] = 4.0f } },
            schema, loaded, "regression", "Y");
        var r3 = service.Predict(new[] { new Dictionary<string, object> { ["X"] = 5.0f } },
            schema, loaded, "regression", "Y");

        Assert.Equal(r1.Rows[0].Score, r2.Rows[0].Score);
        Assert.NotEqual(r1.Rows[0].Score, r3.Rows[0].Score);
    }

    [Fact]
    public void Predict_WithNullTransformer_Throws()
    {
        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" }
            },
            CapturedAt = DateTime.UtcNow
        };

        var service = new PredictionService(_mlContext);
        Assert.Throws<ArgumentNullException>(() =>
            service.Predict(
                new[] { new Dictionary<string, object> { ["X"] = 1.0f } },
                schema,
                (ITransformer)null!,
                "regression",
                "Y"));
    }

    #endregion

    #region MapCategoricalValues

    [Fact]
    public void MapCategoricalValues_ReplacesUnknownValues()
    {
        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new()
                {
                    Name = "Color",
                    DataType = "Categorical",
                    Purpose = "Feature",
                    CategoricalValues = new List<string> { "Red", "Green", "Blue" }
                },
                new() { Name = "Value", DataType = "Numeric", Purpose = "Feature" }
            },
            CapturedAt = DateTime.UtcNow
        };

        var rows = new[]
        {
            new Dictionary<string, object> { ["Color"] = "Red", ["Value"] = 1.0f },
            new Dictionary<string, object> { ["Color"] = "Yellow", ["Value"] = 2.0f }, // unknown
            new Dictionary<string, object> { ["Color"] = "Green", ["Value"] = 3.0f },
        };

        var warnings = new List<string>();
        PredictionService.MapCategoricalValues(rows, schema, warnings);

        Assert.Equal("Red", rows[0]["Color"]);    // unchanged
        Assert.Equal("Red", rows[1]["Color"]);     // replaced with first (most frequent)
        Assert.Equal("Green", rows[2]["Color"]);   // unchanged
        Assert.Single(warnings);
        Assert.Contains("Yellow", warnings[0]);
    }

    [Fact]
    public void MapCategoricalValues_IgnoresNullAndEmpty()
    {
        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new()
                {
                    Name = "Color",
                    DataType = "Categorical",
                    Purpose = "Feature",
                    CategoricalValues = new List<string> { "Red", "Green" }
                }
            },
            CapturedAt = DateTime.UtcNow
        };

        var rows = new[]
        {
            new Dictionary<string, object> { ["Color"] = "" },
            new Dictionary<string, object> { ["Value"] = 1.0f }, // no Color key
        };

        var warnings = new List<string>();
        PredictionService.MapCategoricalValues(rows, schema, warnings);

        Assert.Empty(warnings);
    }

    #endregion

    #region BuildTextLoaderOptions

    [Fact]
    public void BuildTextLoaderOptions_ExcludesExcludedColumns()
    {
        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Removed", DataType = "Numeric", Purpose = "Exclude" },
                new() { Name = "Y", DataType = "Numeric", Purpose = "Label" }
            },
            CapturedAt = DateTime.UtcNow
        };

        var service = new PredictionService(_mlContext);
        var options = service.BuildTextLoaderOptions(schema, "regression", "Y");

        Assert.Equal(2, options.Columns.Length);
        Assert.Equal("X", options.Columns[0].Name);
        Assert.Equal("Y", options.Columns[1].Name);
    }

    [Fact]
    public void BuildTextLoaderOptions_MulticlassLabel_AlwaysString()
    {
        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Label", DataType = "Boolean", Purpose = "Label" }
            },
            CapturedAt = DateTime.UtcNow
        };

        var service = new PredictionService(_mlContext);
        var options = service.BuildTextLoaderOptions(schema, "multiclass-classification", "Label");

        var labelCol = options.Columns.First(c => c.Name == "Label");
        Assert.Equal(DataKind.String, labelCol.DataKind);
    }

    [Fact]
    public void BuildTextLoaderOptions_RegressionBooleanLabel_IsSingle()
    {
        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Y", DataType = "Boolean", Purpose = "Label" }
            },
            CapturedAt = DateTime.UtcNow
        };

        var service = new PredictionService(_mlContext);
        var options = service.BuildTextLoaderOptions(schema, "regression", "Y");

        var labelCol = options.Columns.First(c => c.Name == "Y");
        Assert.Equal(DataKind.Single, labelCol.DataKind);
    }

    #endregion

    #region Helper Classes

    private class SimpleRegression
    {
        public float X { get; set; }
        public float Y { get; set; }
    }

    #endregion
}
