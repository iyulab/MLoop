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

    [Fact]
    public void Predict_Regression_WithInterval_WrapsScoreInConformalBand()
    {
        // ② regression wave: when a RegressionInterval is passed (the caller reads it from the
        // stored conformal band), each regression row exposes [Score - q, Score + q] + confidence.
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
        var rows = new[] { new Dictionary<string, object> { ["X"] = 6.0f } };

        var service = new PredictionService(_mlContext);
        var interval = new RegressionInterval(HalfWidth: 1.5, Confidence: 0.90);
        var result = service.Predict(rows, schema, modelPath, "regression", "Y", interval);

        var row = Assert.Single(result.Rows);
        Assert.NotNull(row.Score);
        Assert.NotNull(row.ScoreLowerBound);
        Assert.NotNull(row.ScoreUpperBound);
        Assert.Equal(row.Score!.Value - 1.5, row.ScoreLowerBound!.Value, 6);
        Assert.Equal(row.Score!.Value + 1.5, row.ScoreUpperBound!.Value, 6);
        Assert.Equal(0.90, row.IntervalConfidence);
    }

    [Fact]
    public void Predict_Regression_WithoutInterval_LeavesBoundsNull()
    {
        // Backward-compatible: no interval passed => point estimate only, bounds stay null.
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
        var rows = new[] { new Dictionary<string, object> { ["X"] = 6.0f } };

        var service = new PredictionService(_mlContext);
        var result = service.Predict(rows, schema, modelPath, "regression", "Y");

        var row = Assert.Single(result.Rows);
        Assert.NotNull(row.Score);
        Assert.Null(row.ScoreLowerBound);
        Assert.Null(row.ScoreUpperBound);
        Assert.Null(row.IntervalConfidence);
    }

    [Fact]
    public void RegressionInterval_FromMetrics_RegressionWithBand_ReturnsPrimaryInterval()
    {
        // Single source both serve and CLI predict use — surfaces the 90% band from stored metrics.
        var metrics = new Dictionary<string, double>
        {
            ["r_squared"] = 0.9,
            ["interval_half_width_80"] = 1.0,
            ["interval_half_width_90"] = 1.5,
            ["interval_half_width_95"] = 2.0,
        };

        var interval = RegressionInterval.FromMetrics("regression", metrics);

        Assert.NotNull(interval);
        Assert.Equal(1.5, interval!.HalfWidth, 6);   // the 90% band, not 80/95
        Assert.Equal(0.90, interval.Confidence, 6);
    }

    [Fact]
    public void RegressionInterval_FromMetrics_NonRegressionOrNoBand_ReturnsNull()
    {
        var bandMetrics = new Dictionary<string, double> { ["interval_half_width_90"] = 1.5 };
        var noBandMetrics = new Dictionary<string, double> { ["r_squared"] = 0.9 };

        Assert.Null(RegressionInterval.FromMetrics("binary-classification", bandMetrics)); // wrong task
        Assert.Null(RegressionInterval.FromMetrics("regression", noBandMetrics));          // no band key
        Assert.Null(RegressionInterval.FromMetrics("regression", null));                   // no metrics
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
    public void BuildTextLoaderOptions_MulticlassCategoricalLabel_IsString()
    {
        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Label", DataType = "Categorical", Purpose = "Label" }
            },
            CapturedAt = DateTime.UtcNow
        };

        var service = new PredictionService(_mlContext);
        var options = service.BuildTextLoaderOptions(schema, "multiclass-classification", "Label");

        var labelCol = options.Columns.First(c => c.Name == "Label");
        Assert.Equal(DataKind.String, labelCol.DataKind);
    }

    [Fact]
    public void BuildTextLoaderOptions_MulticlassNumericLabel_IsSingle()
    {
        // D14: when a multiclass label happens to be numeric-looking (e.g. KAMP SEQ001 class ids
        // "0"/"1"/"2"), train-time schema inference records DataType=Numeric and AutoML's fitted
        // pipeline embeds a MapValueToKey over a Single-typed column, not String. Forcing String
        // here (pre-D14 behavior) mismatches the model's own trained schema — serve /predict throws
        // "Could not apply a map over type 'Single' to column '...' since it has type 'String'".
        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Label", DataType = "Numeric", Purpose = "Label" }
            },
            CapturedAt = DateTime.UtcNow
        };

        var service = new PredictionService(_mlContext);
        var options = service.BuildTextLoaderOptions(schema, "multiclass-classification", "Label");

        var labelCol = options.Columns.First(c => c.Name == "Label");
        Assert.Equal(DataKind.Single, labelCol.DataKind);
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

    #region HasKeyValues (F-21 — clustering predict crash)

    [Fact]
    public void HasKeyValues_KeyColumnWithoutAnnotation_ReturnsFalse()
    {
        // Clustering's KMeans emits PredictedLabel as a bare key (the cluster id) with no KeyValues
        // mapping. Applying MapKeyToValue to it throws "Metadata KeyValues does not exist" — the F-21
        // crash. A key created straight from a [KeyType] field carries no KeyValues annotation.
        var dv = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new KeyOnlyRow { PredictedLabel = 1 },
            new KeyOnlyRow { PredictedLabel = 2 }
        });

        Assert.False(PredictionService.HasKeyValues(dv.Schema["PredictedLabel"]));
    }

    [Fact]
    public void HasKeyValues_KeyColumnWithMapping_ReturnsTrue()
    {
        // A classification PredictedLabel comes from MapValueToKey, which DOES attach KeyValues so the
        // ids can be mapped back to the original label strings — MapKeyToValue is valid there.
        var dv = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new LabelRow { Label = "cat" },
            new LabelRow { Label = "dog" }
        });
        var keyed = _mlContext.Transforms.Conversion
            .MapValueToKey("PredictedLabel", "Label").Fit(dv).Transform(dv);

        Assert.True(PredictionService.HasKeyValues(keyed.Schema["PredictedLabel"]));
    }

    #endregion

    #region Anomaly Detection (D8: Features-vector input)

    [Fact]
    public void Predict_AnomalyDetection_ConcatenatesIndividualColumnsIntoFeatures()
    {
        // Reproduces the real anomaly training shape: data loaded as a single "Features" vector, so the
        // saved RandomizedPca model expects a "Features" input column. The serve path (PredictionService)
        // loads the schema's individual named columns instead — before the D8 fix, model.Transform threw
        // "Could not find input column 'Features'" and serve /predict returned 500 for every anomaly model.
        var trainData = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new AnomalyFeaturesRow { Features = new[] { 10f, 40f, 7f } },
            new AnomalyFeaturesRow { Features = new[] { 11f, 41f, 8f } },
            new AnomalyFeaturesRow { Features = new[] { 9f, 39f, 6f } },
            new AnomalyFeaturesRow { Features = new[] { 12f, 42f, 9f } },
            new AnomalyFeaturesRow { Features = new[] { 8f, 38f, 5f } },
            new AnomalyFeaturesRow { Features = new[] { 10.5f, 40.5f, 7.5f } },
        });
        var pipeline = _mlContext.Transforms.Concatenate("Features", "Features")
            .Append(_mlContext.AnomalyDetection.Trainers.RandomizedPca(
                featureColumnName: "Features", rank: 1, oversampling: 2));
        var model = pipeline.Fit(trainData);

        // Schema as captured in config.InputSchema: individual named numeric feature columns.
        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "pH", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Temp", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Current", DataType = "Numeric", Purpose = "Feature" },
            },
            CapturedAt = DateTime.UtcNow
        };
        var rows = new[]
        {
            new Dictionary<string, object> { ["pH"] = 10.0f, ["Temp"] = 40.0f, ["Current"] = 7.0f },
        };

        var service = new PredictionService(_mlContext);
        var result = service.Predict(rows, schema, model, "anomaly-detection");

        Assert.Equal("anomaly-detection", result.TaskType);
        Assert.Single(result.Rows);
        Assert.NotNull(result.Rows[0].AnomalyScore);
    }

    #endregion

    #region Binary classification (D12/D13 — serve /predict Features vector + Boolean PredictedLabel)

    [Fact]
    public void Predict_Binary_BuildsFeaturesAndReadsBooleanLabel()
    {
        // Reproduces the serve /predict binary failures (honeai-sim campaign): AutoML-style binary
        // models expect a single "Features" vector input (D12) and output PredictedLabel as a raw
        // Boolean (D13). PredictionService loads named scalar columns, so it must (a) build "Features"
        // for classification, and (b) read the Boolean PredictedLabel — otherwise ExtractClassificationRows
        // throws "Invalid TValue: ReadOnlyMemory<Char>, expected Boolean" on every binary model.
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new SimpleBinary { X1 = 1f, X2 = 1f, Label = true },
            new SimpleBinary { X1 = 1.1f, X2 = 0.9f, Label = true },
            new SimpleBinary { X1 = 0.9f, X2 = 1.1f, Label = true },
            new SimpleBinary { X1 = 5f, X2 = 5f, Label = false },
            new SimpleBinary { X1 = 5.1f, X2 = 4.9f, Label = false },
            new SimpleBinary { X1 = 4.9f, X2 = 5.1f, Label = false },
        });

        // Pre-featurize into "Features" and Fit ONLY the trainer, so the saved model expects a "Features"
        // vector input WITHOUT an embedded Concatenate — matching how AutoML's InferColumns loads features
        // (the exact shape that made serve, which loads named columns, fail).
        var featurized = _mlContext.Transforms.Concatenate("Features", "X1", "X2").Fit(data).Transform(data);
        var trainer = _mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
            labelColumnName: "Label", featureColumnName: "Features");
        var model = trainer.Fit(featurized);
        var modelPath = SaveModel(model, featurized.Schema);

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X1", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "X2", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Label", DataType = "Categorical", Purpose = "Label" },
            },
            CapturedAt = DateTime.UtcNow
        };

        var rows = new[]
        {
            new Dictionary<string, object> { ["X1"] = 1.0, ["X2"] = 1.0 },
            new Dictionary<string, object> { ["X1"] = 5.0, ["X2"] = 5.0 },
        };

        var service = new PredictionService(_mlContext);
        var result = service.Predict(rows, schema, modelPath, "binary-classification", "Label");

        Assert.Equal("binary-classification", result.TaskType);
        Assert.Equal(2, result.Rows.Count);
        // PredictedLabel is the raw Boolean rendered as a string (no crash).
        Assert.All(result.Rows, r => Assert.Contains(r.PredictedLabel, new[] { "True", "False" }));
        // Both class probabilities are exposed so max-probability confidence is correct for either class.
        foreach (var r in result.Rows)
        {
            Assert.NotNull(r.Probabilities);
            Assert.True(r.Probabilities!.ContainsKey("True"), "expected 'True' probability");
            Assert.True(r.Probabilities!.ContainsKey("False"), "expected 'False' probability");
            Assert.InRange(r.Probabilities["True"] + r.Probabilities["False"], 0.99, 1.01);
        }
    }

    #endregion

    #region Multiclass classification (D14 — serve /predict numeric-looking class label)

    [Fact]
    public void Predict_Multiclass_NumericLabel_MatchesTrainedSingleSchema()
    {
        // Reproduces the live serve /predict crash found dogfooding KAMP SEQ001 (honeai-sim Wave 3,
        // multiclass label values "0"/"1"/"2"): train-time schema inference records DataType=Numeric
        // for the label, so AutoML's fitted pipeline maps a Single-typed column into a key internally.
        // The pre-D14 serve path always loaded the label as String, so model.Transform threw
        // "Could not apply a map over type 'Single' to column 'Label' since it has type 'String'".
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new SimpleMulticlassNumericLabel { X1 = 1f, X2 = 1f, Label = 0f },
            new SimpleMulticlassNumericLabel { X1 = 1.1f, X2 = 0.9f, Label = 0f },
            new SimpleMulticlassNumericLabel { X1 = 5f, X2 = 5f, Label = 1f },
            new SimpleMulticlassNumericLabel { X1 = 5.1f, X2 = 4.9f, Label = 1f },
            new SimpleMulticlassNumericLabel { X1 = 9f, X2 = 9f, Label = 2f },
            new SimpleMulticlassNumericLabel { X1 = 9.1f, X2 = 8.9f, Label = 2f },
        });

        var keyed = _mlContext.Transforms.Conversion.MapValueToKey("Label", "Label").Fit(data).Transform(data);
        var featurized = _mlContext.Transforms.Concatenate("Features", "X1", "X2").Fit(keyed).Transform(keyed);
        var trainer = _mlContext.MulticlassClassification.Trainers.LightGbm(
            labelColumnName: "Label", featureColumnName: "Features");
        var model = trainer.Fit(featurized);
        var modelPath = SaveModel(model, featurized.Schema);

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X1", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "X2", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Label", DataType = "Numeric", Purpose = "Label" },
            },
            CapturedAt = DateTime.UtcNow
        };

        var rows = new[]
        {
            new Dictionary<string, object> { ["X1"] = 1.0, ["X2"] = 1.0 },
            new Dictionary<string, object> { ["X1"] = 9.0, ["X2"] = 9.0 },
        };

        var service = new PredictionService(_mlContext);
        var result = service.Predict(rows, schema, modelPath, "multiclass-classification", "Label");

        Assert.Equal("multiclass-classification", result.TaskType);
        Assert.Equal(2, result.Rows.Count);
        Assert.All(result.Rows, r => Assert.Contains(r.PredictedLabel, new[] { "0", "1", "2" }));
    }

    #endregion

    #region Helper Classes

    private class SimpleRegression
    {
        public float X { get; set; }
        public float Y { get; set; }
    }

    private class SimpleBinary
    {
        public float X1 { get; set; }
        public float X2 { get; set; }
        public bool Label { get; set; }
    }

    private class SimpleMulticlassNumericLabel
    {
        public float X1 { get; set; }
        public float X2 { get; set; }
        public float Label { get; set; }
    }

    private class AnomalyFeaturesRow
    {
        [VectorType(3)]
        public float[] Features { get; set; } = new float[3];
    }

    private class KeyOnlyRow
    {
        [KeyType(1000)]
        public uint PredictedLabel { get; set; }
    }

    private class LabelRow
    {
        public string Label { get; set; } = "";
    }

    #endregion
}
