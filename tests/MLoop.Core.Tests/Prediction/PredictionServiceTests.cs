using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.AutoML;
using MLoop.Core.Models;
using MLoop.Core.Prediction;

namespace MLoop.Core.Tests.Prediction;

public class PredictionServiceTests : IDisposable
{
    private readonly MLContext _mlContext = new(seed: 42);
    private readonly List<string> _tempFiles = new();
    private readonly List<string> _tempDirs = new();

    public void Dispose()
    {
        foreach (var file in _tempFiles)
        {
            try { File.Delete(file); } catch { }
        }
        foreach (var dir in _tempDirs)
        {
            try { Directory.Delete(dir, recursive: true); } catch { }
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
        // ConfidencePolicy post-pass populates the normalized per-row confidence from the band.
        Assert.NotNull(row.Confidence);
        Assert.InRange(row.Confidence!.Value, 0.0, 1.0);
    }

    [Fact]
    public void Predict_Regression_Heteroscedastic_ProducesPerRowBandWidths()
    {
        // ② regression wave (heteroscedastic, serve path): the σ-model widens the band per row. Build
        // data whose residual magnitude grows with X, fit the real normalized-conformal aux, save both
        // models side by side, and assert two inputs with very different X get different band widths.
        // Dedicated context: SDCA is RNG-sensitive and the shared _mlContext is advanced by other tests,
        // which would make this model-quality-dependent assertion order-flaky. A fresh seeded context
        // makes the σ-model fit deterministic regardless of suite order.
        var ml = new MLContext(seed: 42);
        var rows = Enumerable.Range(0, 60).Select(i =>
        {
            float x = (i % 20) + 1;
            return new SimpleRegression { X = x, Y = 2f * x + 0.5f * x * ((i % 2 == 0) ? 1 : -1) }; // |resid from 2X| = 0.5X
        }).ToList();
        var data = ml.Data.LoadFromEnumerable(rows);
        var mainModel = ml.Transforms.Concatenate("Features", "X")
            .Append(ml.Regression.Trainers.Sdca(labelColumnName: "Y")).Fit(data);
        var scored = mainModel.Transform(data);

        var norm = AutoMLRunner.ComputeNormalizedConformal(ml, scored, "Y", new[] { "Features" });
        Assert.NotNull(norm);
        var metrics = new Dictionary<string, double>(AutoMLRunner.ComputeConformalIntervals(scored, "Y"));
        foreach (var kv in norm!.Metrics) metrics[kv.Key] = kv.Value;
        var interval = RegressionInterval.FromMetrics("regression", metrics);
        Assert.NotNull(interval);
        Assert.True(interval!.IsHeteroscedastic);

        var dir = Path.Combine(Path.GetTempPath(), "mloop-hetero-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        _tempDirs.Add(dir);
        ml.Model.Save(mainModel, data.Schema, Path.Combine(dir, "model.zip"));
        ml.Model.Save(norm.AuxModel, null, Path.Combine(dir, "residual-model.zip"));

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Y", DataType = "Numeric", Purpose = "Label" }
            },
            CapturedAt = DateTime.UtcNow
        };
        var inputRows = new[]
        {
            new Dictionary<string, object> { ["X"] = 1.0f },
            new Dictionary<string, object> { ["X"] = 20.0f },
        };

        var service = new PredictionService(ml);
        var result = service.Predict(inputRows, schema, Path.Combine(dir, "model.zip"), "regression", "Y", interval);

        Assert.Equal(2, result.Rows.Count);
        double w1 = result.Rows[0].ScoreUpperBound!.Value - result.Rows[0].ScoreLowerBound!.Value;
        double w2 = result.Rows[1].ScoreUpperBound!.Value - result.Rows[1].ScoreLowerBound!.Value;

        // Diagnostic readout of the aux σ-model itself: on a failure this tells apart "the σ-model fit
        // degenerated to a constant" from "the service path fell back to the constant-width band"
        // (the macOS-arm64 CI failure class — see ISSUE-mloop-20260705-macos-predictionservice-test-failures).
        var probe = ml.Data.LoadFromEnumerable(new[]
        {
            new SimpleRegression { X = 1.0f },
            new SimpleRegression { X = 20.0f },
        });
        var probeSigmas = norm.AuxModel.Transform(mainModel.Transform(probe)).GetColumn<float>("Score").ToArray();

        Assert.True(Math.Abs(w1 - w2) > 1e-3,
            $"per-row band widths should differ (w1={w1:F3}, w2={w2:F3}; aux σ(X=1)={probeSigmas[0]:F4}, σ(X=20)={probeSigmas[1]:F4})");
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
        Assert.False(interval.IsHeteroscedastic);            // no norm keys → constant width
        Assert.Equal(1.5, interval.WidthFor(99.0), 6);       // width ignores σ in constant mode
    }

    [Fact]
    public void RegressionInterval_FromMetrics_WithNormalizedBand_ReturnsHeteroscedastic()
    {
        // ② regression wave (heteroscedastic): when the normalized-conformal keys are present the band
        // becomes per-row — width = q·(max(σ,0)+β) — and keeps the constant half-width as its fallback.
        var metrics = new Dictionary<string, double>
        {
            ["interval_half_width_90"] = 1.5,   // constant fallback
            ["norm_interval_q_90"] = 2.0,       // normalized quantile q
            ["interval_beta"] = 0.5,            // σ floor β
        };

        var interval = RegressionInterval.FromMetrics("regression", metrics);

        Assert.NotNull(interval);
        Assert.True(interval!.IsHeteroscedastic);
        Assert.Equal(1.5, interval.HalfWidth, 6);            // fallback preserved
        Assert.Equal(2.0, interval.NormalizedQ!.Value, 6);
        Assert.Equal(0.5, interval.Beta!.Value, 6);
        // width grows with σ: q·(σ+β). σ=1 → 2·1.5=3.0; σ=3 → 2·3.5=7.0; negative σ clamped to 0 → 2·0.5=1.0
        Assert.Equal(3.0, interval.WidthFor(1.0), 6);
        Assert.Equal(7.0, interval.WidthFor(3.0), 6);
        Assert.Equal(1.0, interval.WidthFor(-5.0), 6);
        Assert.True(interval.WidthFor(3.0) > interval.WidthFor(1.0)); // wider where σ larger
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

    // D20 (cycle-146, serve image dogfooding): rows sharing NO column with the trained schema
    // (e.g. an envelope like {"rows":[...]} posted to /predict) used to default every column and
    // return 200 with a fabricated all-zero-score label and no warning. Zero overlap must fail fast.
    [Fact]
    public void Predict_RowsShareNoSchemaColumn_ThrowsActionableArgumentException()
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

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Y", DataType = "Numeric", Purpose = "Label" }
            },
            CapturedAt = DateTime.UtcNow
        };

        var rows = new[] { new Dictionary<string, object> { ["rows"] = "[{\"X\":1}]" } };

        var service = new PredictionService(_mlContext);
        var ex = Assert.Throws<ArgumentException>(
            () => service.Predict(rows, schema, model, "regression", "Y"));
        Assert.Contains("none of the model's input columns", ex.Message);
        Assert.Contains("X", ex.Message);   // actionable: names the expected columns
    }

    // D21 (cycle-150, forecasting serve dogfooding): posting rows to a forecasting model used to
    // return 200 with every output field null (the stateful SSA forecaster extracts nothing from a
    // stateless per-row Transform — silent no-op, same family as D20). Must fail fast instead,
    // pointing the caller at the horizon-based CLI path. Fires before the model file is touched.
    [Fact]
    public void Predict_Forecasting_RowBased_ThrowsActionableArgumentException()
    {
        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "Temp", DataType = "Numeric", Purpose = "Label" }
            },
            CapturedAt = DateTime.UtcNow
        };
        var rows = new[] { new Dictionary<string, object> { ["Temp"] = 42.0f } };

        var service = new PredictionService(_mlContext);
        // Path overload: guard must reject before attempting to load the (nonexistent) model.
        var ex = Assert.Throws<ArgumentException>(
            () => service.Predict(rows, schema, "nonexistent.zip", "forecasting", "Temp"));
        Assert.Contains("row-based prediction", ex.Message);
        Assert.Contains("mloop predict", ex.Message);   // actionable: names the working path

        // Case-insensitive, and empty rows don't bypass the guard.
        Assert.Throws<ArgumentException>(
            () => service.Predict(Array.Empty<Dictionary<string, object>>(), schema, "nonexistent.zip", "Forecasting", "Temp"));
    }

    // The pre-loaded-transformer overload (serve model cache path) rejects forecasting too.
    [Fact]
    public void Predict_Forecasting_WithPreLoadedTransformer_Throws()
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

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Y", DataType = "Numeric", Purpose = "Label" }
            },
            CapturedAt = DateTime.UtcNow
        };
        var rows = new[] { new Dictionary<string, object> { ["X"] = 1.0f } };

        var service = new PredictionService(_mlContext);
        var ex = Assert.Throws<ArgumentException>(
            () => service.Predict(rows, schema, model, "forecasting", "Y"));
        Assert.Contains("row-based prediction", ex.Message);
    }

    // Partial overlap stays legitimate — missing columns default as before (no regression from D20).
    [Fact]
    public void Predict_RowsWithPartialSchemaOverlap_StillPredicts()
    {
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new TwoFeatureRegression { X = 1.0f, Z = 1.0f, Y = 2.0f },
            new TwoFeatureRegression { X = 2.0f, Z = 1.0f, Y = 4.0f },
            new TwoFeatureRegression { X = 3.0f, Z = 1.0f, Y = 6.0f },
        });
        var pipeline = _mlContext.Transforms.Concatenate("Features", "X", "Z")
            .Append(_mlContext.Regression.Trainers.Sdca(labelColumnName: "Y"));
        var model = pipeline.Fit(data);

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Z", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Y", DataType = "Numeric", Purpose = "Label" }
            },
            CapturedAt = DateTime.UtcNow
        };

        // "Z" missing — defaults; must not throw.
        var rows = new[] { new Dictionary<string, object> { ["X"] = 2.0f } };

        var service = new PredictionService(_mlContext);
        var result = service.Predict(rows, schema, model, "regression", "Y");
        Assert.Single(result.Rows);
        Assert.NotNull(result.Rows[0].Score);
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

    // D22 (cycle-152, TS-anomaly live dogfooding): time-series-anomaly models emit a single vector
    // column (Prediction = [alert, raw score, ...]) instead of PredictedLabel/Score, so the generic
    // extraction read none of it — serve /predict and mloop predict --json returned rows of all-null
    // fields with 200/exit-0 (observed live: 5000 × {} on a KAMP sensor series). The dedicated
    // extractor must surface the alert and the raw score.
    /// <summary>
    /// SrCnn's spectral-residual FFT goes through MKL (Microsoft.ML.Transforms.TimeSeries.FftUtils →
    /// MklImports/libiomp5), which is not present on every CI runner (observed: linux-x64 GitHub
    /// runner). Probe once and skip the real-SrCnn round-trip where MKL is absent — the same guard
    /// pattern as ObjectDetectionEvaluatorTests (libtorch) and PredictForecastingTests (SSA).
    /// </summary>
    private static readonly Lazy<bool> MklAvailable = new(() =>
    {
        try
        {
            var ml = new MLContext(seed: 0);
            var data = ml.Data.LoadFromEnumerable(
                Enumerable.Range(0, 16).Select(i => new SimpleSeries { Value = i }));
            var transformed = ml.Transforms.DetectAnomalyBySrCnn(
                    outputColumnName: TimeSeriesAnomalyOutput.PredictionColumnName,
                    inputColumnName: nameof(SimpleSeries.Value),
                    windowSize: 8, backAddWindowSize: 5, lookaheadWindowSize: 5,
                    averagingWindowSize: 3, judgementWindowSize: 8, threshold: 0.3)
                .Fit(data).Transform(data);
            // Materialize EVERY row: the FFT (and hence the MKL load) only fires once the sliding
            // window fills, so previewing a single warmup row would falsely report MKL as present.
            using var cursor = transformed.GetRowCursor(transformed.Schema);
            var col = transformed.Schema[TimeSeriesAnomalyOutput.PredictionColumnName];
            var getter = cursor.GetGetter<VBuffer<double>>(col);
            VBuffer<double> buf = default;
            while (cursor.MoveNext())
                getter(ref buf);
            return true;
        }
        catch (Exception ex) when (ex is DllNotFoundException or TypeInitializationException
                                   || ex.InnerException is DllNotFoundException)
        {
            return false;
        }
    });

    [Fact]
    public void Predict_TimeSeriesAnomaly_ExtractsAlertAndRawScore()
    {
        if (!MklAvailable.Value)
            return; // MKL natives absent (e.g. linux CI); integration coverage runs where MKL exists.

        // Flat-ish series with one large spike — SrCnn (the campaign's winning detector) must alert on it.
        var series = new List<SimpleSeries>();
        for (int i = 0; i < 40; i++)
            series.Add(new SimpleSeries { Value = 10f + (i % 3) * 0.1f });
        series[30] = new SimpleSeries { Value = 100f };

        var trainData = _mlContext.Data.LoadFromEnumerable(series);
        var pipeline = _mlContext.Transforms.DetectAnomalyBySrCnn(
            outputColumnName: TimeSeriesAnomalyOutput.PredictionColumnName,
            inputColumnName: "Value",
            windowSize: 8,
            backAddWindowSize: 5,
            lookaheadWindowSize: 5,
            averagingWindowSize: 3,
            judgementWindowSize: 8,   // must be <= windowSize (ML.NET contract)
            threshold: 0.3);
        var model = pipeline.Fit(trainData);

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "Value", DataType = "Numeric", Purpose = "Feature" },
            },
            CapturedAt = DateTime.UtcNow
        };
        var rows = series.Select(s => new Dictionary<string, object> { ["Value"] = s.Value }).ToArray();

        var service = new PredictionService(_mlContext);
        var result = service.Predict(rows, schema, model, "time-series-anomaly");

        Assert.Equal("time-series-anomaly", result.TaskType);
        Assert.Equal(series.Count, result.Rows.Count);
        Assert.All(result.Rows, r =>
        {
            Assert.NotNull(r.IsAnomaly);      // alert slot surfaced
            Assert.NotNull(r.AnomalyScore);   // raw-score slot surfaced
        });
        Assert.Contains(result.Rows, r => r.IsAnomaly == true);   // the spike alerts
        // The raw score is NOT a [0,1] boundary-0.5 probability — no confidence may be fabricated
        // from it until the TS-anomaly signal mapping is designed (P3-3).
        Assert.All(result.Rows, r => Assert.Null(r.Confidence));
    }

    // D26 (cycle-155, ranking live dogfooding): ranking was missing from RequiresFeaturesVectorInput,
    // so the loaded named columns never got a "Features" vector and every structured ranking predict
    // (serve /predict, mloop predict --json) failed with "Could not find input column 'Features'"
    // while the CSV path worked. Mirrors the RunRankingAsync model shape: embedded ConvertType +
    // MapValueToKey(GroupId) + Concatenate + ranking trainer.
    [Fact]
    public void Predict_Ranking_BuildsFeaturesAndReturnsScores()
    {
        var data = new List<RankingRow>();
        var rng = new Random(7);
        for (int g = 0; g < 5; g++)
            for (int i = 0; i < 20; i++)
            {
                var f1 = (float)rng.NextDouble();
                var f2 = (float)rng.NextDouble();
                data.Add(new RankingRow
                {
                    Query = $"q{g}",
                    F1 = f1,
                    F2 = f2,
                    Label = (float)Math.Clamp(Math.Round(2 * (0.7 * f1 + 0.3 * f2)), 0, 2)
                });
            }
        var trainData = _mlContext.Data.LoadFromEnumerable(data);

        var pipeline = _mlContext.Transforms.Conversion.ConvertType("Label", outputKind: DataKind.Single)
            .Append(_mlContext.Transforms.Conversion.MapValueToKey("GroupId", "Query"))
            .Append(_mlContext.Transforms.Concatenate("Features", "F1", "F2"))
            .Append(_mlContext.Ranking.Trainers.FastTree(
                labelColumnName: "Label", featureColumnName: "Features", rowGroupColumnName: "GroupId",
                numberOfTrees: 5, numberOfLeaves: 4, minimumExampleCountPerLeaf: 2));
        var model = pipeline.Fit(trainData);

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "Query", DataType = "Categorical", Purpose = "Feature" },
                new() { Name = "F1", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "F2", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Label", DataType = "Numeric", Purpose = "Label" },
            },
            CapturedAt = DateTime.UtcNow
        };
        var rows = new[]
        {
            new Dictionary<string, object> { ["Query"] = "q0", ["F1"] = 0.9f, ["F2"] = 0.8f },
            new Dictionary<string, object> { ["Query"] = "q0", ["F1"] = 0.1f, ["F2"] = 0.2f },
        };

        var service = new PredictionService(_mlContext);
        var result = service.Predict(rows, schema, model, "ranking", "Label");

        Assert.Equal(2, result.Rows.Count);
        Assert.All(result.Rows, r => Assert.NotNull(r.Score));
        // Higher-feature row must outrank the lower one (sanity: the score is a real ranking score).
        Assert.True(result.Rows[0].Score > result.Rows[1].Score);
        // D25: a ranking score is not a [0,1] confidence — nothing may be fabricated from it.
        Assert.All(result.Rows, r => Assert.Null(r.Confidence));
    }

    #endregion

    // P-svc1 (cycle-159): generalizes D20~D26 — a task/model mismatch (a model trained for one task
    // used with a different declared taskType) can leave every row's defining output field null while
    // extraction still returns a row per input and the caller still gets 200/success. That silent
    // all-null result is indistinguishable from "nothing to report". These tests force the mismatch
    // directly (rather than reproducing a specific historical bug) to pin the generalized backstop.
    #region P-svc1 (output-contract validation — task/model mismatch)

    [Fact]
    public void Predict_RegressionModelDeclaredAsMulticlass_ThrowsActionableException()
    {
        // A regression model's scored schema has "Score" but no "PredictedLabel" at all — declaring it
        // as multiclass-classification leaves PredictedLabel null on every row.
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new SimpleRegression { X = 1.0f, Y = 2.0f },
            new SimpleRegression { X = 2.0f, Y = 4.0f },
            new SimpleRegression { X = 3.0f, Y = 6.0f },
        });
        var pipeline = _mlContext.Transforms.Concatenate("Features", "X")
            .Append(_mlContext.Regression.Trainers.Sdca(labelColumnName: "Y"));
        var model = pipeline.Fit(data);

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Y", DataType = "Numeric", Purpose = "Label" }
            },
            CapturedAt = DateTime.UtcNow
        };
        var rows = new[] { new Dictionary<string, object> { ["X"] = 1.0f } };

        var service = new PredictionService(_mlContext);
        var ex = Assert.Throws<DegeneratePredictionException>(
            () => service.Predict(rows, schema, model, "multiclass-classification", "Y"));
        Assert.Contains("multiclass-classification", ex.Message);
        Assert.Contains("every defining output", ex.Message);
    }

    [Fact]
    public void Predict_RankingModelDeclaredAsClustering_ThrowsActionableException()
    {
        // A ranking model's scored schema has "Score" but no "PredictedLabel" (no cluster id key) —
        // declaring it as clustering leaves ClusterId null on every row.
        var data = new List<RankingRow>();
        var rng = new Random(7);
        for (int g = 0; g < 3; g++)
            for (int i = 0; i < 10; i++)
                data.Add(new RankingRow { Query = $"q{g}", F1 = (float)rng.NextDouble(), F2 = (float)rng.NextDouble(), Label = i % 3 });
        var trainData = _mlContext.Data.LoadFromEnumerable(data);

        var pipeline = _mlContext.Transforms.Conversion.ConvertType("Label", outputKind: DataKind.Single)
            .Append(_mlContext.Transforms.Conversion.MapValueToKey("GroupId", "Query"))
            .Append(_mlContext.Transforms.Concatenate("Features", "F1", "F2"))
            .Append(_mlContext.Ranking.Trainers.FastTree(
                labelColumnName: "Label", featureColumnName: "Features", rowGroupColumnName: "GroupId",
                numberOfTrees: 5, numberOfLeaves: 4, minimumExampleCountPerLeaf: 2));
        var model = pipeline.Fit(trainData);

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "Query", DataType = "Categorical", Purpose = "Feature" },
                new() { Name = "F1", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "F2", DataType = "Numeric", Purpose = "Feature" },
                new() { Name = "Label", DataType = "Numeric", Purpose = "Label" },
            },
            CapturedAt = DateTime.UtcNow
        };
        var rows = new[] { new Dictionary<string, object> { ["Query"] = "q0", ["F1"] = 0.5f, ["F2"] = 0.5f } };

        var service = new PredictionService(_mlContext);
        var ex = Assert.Throws<DegeneratePredictionException>(
            () => service.Predict(rows, schema, model, "clustering", "Label"));
        Assert.Contains("clustering", ex.Message);
    }

    [Fact]
    public void Predict_FeaturizeOnlyModelDeclaredAsAnomalyDetection_ThrowsActionableException()
    {
        // A pure featurizer (no trainer appended) has neither "PredictedLabel" nor "Score" — declaring
        // it as anomaly-detection leaves both IsAnomaly and AnomalyScore null on every row.
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new SimpleRegression { X = 1.0f, Y = 2.0f },
            new SimpleRegression { X = 2.0f, Y = 4.0f },
        });
        var model = _mlContext.Transforms.Concatenate("Features", "X").Fit(data);

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" },
            },
            CapturedAt = DateTime.UtcNow
        };
        var rows = new[] { new Dictionary<string, object> { ["X"] = 1.0f } };

        var service = new PredictionService(_mlContext);
        var ex = Assert.Throws<DegeneratePredictionException>(
            () => service.Predict(rows, schema, model, "anomaly-detection"));
        Assert.Contains("anomaly-detection", ex.Message);
    }

    [Fact]
    public void Predict_FeaturizeOnlyModelDeclaredAsTimeSeriesAnomaly_ThrowsActionableException()
    {
        // Same featurizer-only model as above, declared as time-series-anomaly instead — neither
        // "Prediction" nor a fallback Score/PredictedLabel exists, so the TS-anomaly extractor also
        // comes back fully null.
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new SimpleRegression { X = 1.0f, Y = 2.0f },
            new SimpleRegression { X = 2.0f, Y = 4.0f },
        });
        var model = _mlContext.Transforms.Concatenate("Features", "X").Fit(data);

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema>
            {
                new() { Name = "X", DataType = "Numeric", Purpose = "Feature" },
            },
            CapturedAt = DateTime.UtcNow
        };
        var rows = new[] { new Dictionary<string, object> { ["X"] = 1.0f } };

        var service = new PredictionService(_mlContext);
        var ex = Assert.Throws<DegeneratePredictionException>(
            () => service.Predict(rows, schema, model, "time-series-anomaly"));
        Assert.Contains("time-series-anomaly", ex.Message);
    }

    // No-false-positive check: `Predict_AnomalyDetection_ConcatenatesIndividualColumnsIntoFeatures`
    // below already asserts a legitimate anomaly-detection result (non-null AnomalyScore) passes
    // through — if RequireNonDegenerateOutput false-triggered there, that existing test would fail.

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
        // SdcaMaximumEntropy, not LightGbm: the pinned regression is the serve path's label *loading*
        // (String vs Single before the model's MapValueToKey), so any multiclass trainer reproduces it —
        // and lib_lightgbm ships no osx-arm64 native, which made this test DllNotFound on macOS CI.
        var trainer = _mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
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

    #region ComputeRowConfidences (① — CSV path reuses the single ConfidencePolicy authority)

    [Fact]
    public void ComputeRowConfidences_RegressionWithBandColumns_DerivesFromWrittenBounds()
    {
        // The CLI CSV path applies the conformal band BEFORE extracting confidence, so the scored view
        // already carries ScoreLowerBound/ScoreUpperBound. ComputeRowConfidences must read those actual
        // per-row bounds (not recompute), so the Confidence column stays consistent with the band already
        // in the CSV. residual_std=2, k=3 → 1 - half/(3*2): half=1 → 0.8333, half=3 → 0.5, half=7 → 0.
        var view = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new RegressionBandRow { Score = 15f, ScoreLowerBound = 14f, ScoreUpperBound = 16f }, // narrow
            new RegressionBandRow { Score = 15f, ScoreLowerBound = 12f, ScoreUpperBound = 18f }, // moderate
            new RegressionBandRow { Score = 15f, ScoreLowerBound = 8f, ScoreUpperBound = 22f },  // wide
        });
        var interval = new RegressionInterval(HalfWidth: 99.0, Confidence: 0.90, ResidualStd: 2.0);

        var conf = PredictionService.ComputeRowConfidences(view, "regression", interval);

        Assert.Equal(3, conf.Count);
        Assert.Equal(0.8333, conf[0]!.Value, 3);
        Assert.Equal(0.5000, conf[1]!.Value, 3);
        Assert.Equal(0.0000, conf[2]!.Value, 3);
        // Reads the actual bounds, NOT interval.HalfWidth (99) — else every row would clamp to 0.
    }

    [Fact]
    public void ComputeRowConfidences_Binary_ReturnsMaxClassProbabilityPerRow()
    {
        // Reuses ExtractClassificationRows + ConfidencePolicy over a real scored view, so the CSV path's
        // confidence matches the --json path's exactly (one authority, no re-derivation).
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new SimpleBinary { X1 = 1f, X2 = 1f, Label = true },
            new SimpleBinary { X1 = 1.1f, X2 = 0.9f, Label = true },
            new SimpleBinary { X1 = 0.9f, X2 = 1.1f, Label = true },
            new SimpleBinary { X1 = 5f, X2 = 5f, Label = false },
            new SimpleBinary { X1 = 5.1f, X2 = 4.9f, Label = false },
            new SimpleBinary { X1 = 4.9f, X2 = 5.1f, Label = false },
        });
        var model = _mlContext.Transforms.Concatenate("Features", "X1", "X2")
            .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: "Label", featureColumnName: "Features"))
            .Fit(data);
        var scored = model.Transform(data);

        var conf = PredictionService.ComputeRowConfidences(scored, "binary-classification");

        Assert.Equal(6, conf.Count);
        // Confidence = max(P(True), P(False)) ∈ [0.5, 1] for every row (both classes exposed).
        Assert.All(conf, c =>
        {
            Assert.NotNull(c);
            Assert.InRange(c!.Value, 0.5, 1.0);
        });
    }

    [Fact]
    public void ComputeRowConfidences_DegenerateView_ReturnsNullsWithoutThrowing()
    {
        // ExtractResults throws RequireNonDegenerateOutput on an all-null scored view (D20~D26 guard).
        // ComputeRowConfidences is a read-only enrichment over a view the caller already writes, so it must
        // NOT throw — it returns null confidence per row and lets the caller omit the column.
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new SimpleRegression { X = 1.0f, Y = 2.0f },
            new SimpleRegression { X = 2.0f, Y = 4.0f },
        });
        var featurizeOnly = _mlContext.Transforms.Concatenate("Features", "X").Fit(data);
        var scored = featurizeOnly.Transform(data);

        var conf = PredictionService.ComputeRowConfidences(scored, "anomaly-detection");

        Assert.Equal(2, conf.Count);
        Assert.All(conf, Assert.Null);
    }

    [Fact]
    public void ComputeRowConfidences_RegressionPointOnly_ReturnsNull()
    {
        // A bare regression point estimate (no band columns) carries no confidence signal — null, and the
        // CLI omits the Confidence column entirely (honest: Score is a target value, not a confidence).
        var pointView = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new ScoreOnlyRow { Score = 12.5f },
        });

        var conf = PredictionService.ComputeRowConfidences(pointView, "regression");

        var c = Assert.Single(conf);
        Assert.Null(c);
    }

    #endregion

    // Live repro (2026-07-12, v0.22.0): FastForest fitted on an 8-row set scored NaN for EVERY input;
    // the NaN flowed through as a value — SaveAsText wrote literal '?' into the user's CSV with exit 0
    // and System.Text.Json crashed the --json/serve path. Non-finite signal values are scrubbed to null
    // at the single materialization authority, so the all-degenerate guard catches the all-NaN case and
    // partial rows serialize as honest nulls.
    #region Non-finite score scrub (degenerate NaN-scoring model)

    [Fact]
    public void Predict_AllRowsScoreNonFinite_ThrowsDegeneratePredictionException()
    {
        var fitSource = _mlContext.Data.LoadFromEnumerable(new[] { new XOnlyRow { X = 1f } });
        var model = _mlContext.Transforms.CustomMapping<XOnlyRow, ScoreOnlyRow>(
            (input, output) => output.Score = float.NaN, contractName: null).Fit(fitSource);

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema> { new() { Name = "X", DataType = "Numeric", Purpose = "Feature" } },
            CapturedAt = DateTime.UtcNow
        };
        var rows = new[]
        {
            new Dictionary<string, object> { ["X"] = 1.0f },
            new Dictionary<string, object> { ["X"] = 2.0f },
        };

        var service = new PredictionService(_mlContext);
        var ex = Assert.Throws<DegeneratePredictionException>(
            () => service.Predict(rows, schema, model, "regression"));
        Assert.Contains("non-finite", ex.Message);
    }

    [Fact]
    public void Predict_PartiallyNonFiniteScores_MapsThoseRowsToNullAndSucceeds()
    {
        var fitSource = _mlContext.Data.LoadFromEnumerable(new[] { new XOnlyRow { X = 1f } });
        var model = _mlContext.Transforms.CustomMapping<XOnlyRow, ScoreOnlyRow>(
            (input, output) => output.Score = input.X == 1f ? float.NaN : input.X * 2f,
            contractName: null).Fit(fitSource);

        var schema = new InputSchemaInfo
        {
            Columns = new List<ColumnSchema> { new() { Name = "X", DataType = "Numeric", Purpose = "Feature" } },
            CapturedAt = DateTime.UtcNow
        };
        var rows = new[]
        {
            new Dictionary<string, object> { ["X"] = 1.0f },
            new Dictionary<string, object> { ["X"] = 3.0f },
        };

        var service = new PredictionService(_mlContext);
        var result = service.Predict(rows, schema, model, "regression");

        Assert.Equal(2, result.Rows.Count);
        Assert.Null(result.Rows[0].Score);       // NaN scrubbed to "no point estimate"
        Assert.Null(result.Rows[0].Confidence);
        Assert.NotNull(result.Rows[1].Score);
        Assert.Equal(6.0, result.Rows[1].Score!.Value, 3);
    }

    [Fact]
    public void RequireNonDegenerateScoredView_AllNonFinite_Throws()
    {
        // The CLI CSV seam: SaveAsText serializes the raw IDataView, so this guard is what stands
        // between an all-NaN model and a CSV full of literal '?' values written with exit 0.
        var view = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new ScoreOnlyRow { Score = float.NaN },
            new ScoreOnlyRow { Score = float.PositiveInfinity },
        });

        Assert.Throws<DegeneratePredictionException>(
            () => PredictionService.RequireNonDegenerateScoredView(view, "regression"));
    }

    [Fact]
    public void RequireNonDegenerateScoredView_FiniteScores_Passes()
    {
        var view = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new ScoreOnlyRow { Score = float.NaN },  // one bad row is legitimate
            new ScoreOnlyRow { Score = 3.5f },
        });

        PredictionService.RequireNonDegenerateScoredView(view, "regression"); // must not throw
    }

    [Fact]
    public void ComputeRowConfidences_NonFiniteBandBounds_ReturnsNull()
    {
        // A NaN band is no band: previously half-width = NaN clamped to confidence 0 (fabricated
        // "fully uncertain"); scrubbed bounds mean point-only → honest null → column omitted.
        var view = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new RegressionBandRow { Score = 15f, ScoreLowerBound = float.NaN, ScoreUpperBound = float.NaN },
        });
        var interval = new RegressionInterval(HalfWidth: 1.0, Confidence: 0.90, ResidualStd: 2.0);

        var conf = PredictionService.ComputeRowConfidences(view, "regression", interval);

        var c = Assert.Single(conf);
        Assert.Null(c);
    }

    #endregion

    #region Helper Classes

    private class XOnlyRow
    {
        public float X { get; set; }
    }

    private class SimpleRegression
    {
        public float X { get; set; }
        public float Y { get; set; }
    }

    private class RegressionBandRow
    {
        public float Score { get; set; }
        public float ScoreLowerBound { get; set; }
        public float ScoreUpperBound { get; set; }
    }

    private class ScoreOnlyRow
    {
        public float Score { get; set; }
    }

    private class TwoFeatureRegression
    {
        public float X { get; set; }
        public float Z { get; set; }
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

    private class SimpleSeries
    {
        public float Value { get; set; }
    }

    private class RankingRow
    {
        public string Query { get; set; } = "";
        public float F1 { get; set; }
        public float F2 { get; set; }
        public float Label { get; set; }
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
