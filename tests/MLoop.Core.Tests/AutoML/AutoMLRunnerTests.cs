using System.Linq;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLoop.Core.AutoML;

namespace MLoop.Core.Tests.AutoML;

public class AutoMLRunnerTests
{
    private readonly MLContext _mlContext = new MLContext(seed: 42);

    #region BuildColumnInformation Tests

    [Fact]
    public void BuildColumnInformation_TextOnlyDataset_ReturnsWithTextColumns()
    {
        // Arrange: text-only dataset (Content + Label)
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new TextBinaryData { Content = "This is a positive review", Label = true },
            new TextBinaryData { Content = "This is a negative review", Label = false },
            new TextBinaryData { Content = "Another positive text", Label = true },
        });

        // Act
        var result = AutoMLRunner.BuildColumnInformation(data, "Label");

        // Assert
        Assert.NotNull(result);
        Assert.Equal("Label", result.LabelColumnName);
        Assert.Contains("Content", result.TextColumnNames);
        Assert.Empty(result.NumericColumnNames);
        Assert.Empty(result.IgnoredColumnNames);
    }

    [Fact]
    public void BuildColumnInformation_NumericOnlyDataset_ReturnsNull()
    {
        // Arrange: numeric-only dataset
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new NumericData { Feature1 = 1.0f, Feature2 = 2.0f, Label = true },
            new NumericData { Feature1 = 3.0f, Feature2 = 4.0f, Label = false },
        });

        // Act
        var result = AutoMLRunner.BuildColumnInformation(data, "Label");

        // Assert: null means use existing behavior (no text columns)
        Assert.Null(result);
    }

    [Fact]
    public void BuildColumnInformation_MixedTextAndNumeric_ReturnsBothClassified()
    {
        // Arrange: mixed dataset
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new MixedData { Description = "Good product", Price = 29.99f, Label = true },
            new MixedData { Description = "Bad product", Price = 9.99f, Label = false },
        });

        // Act
        var result = AutoMLRunner.BuildColumnInformation(data, "Label");

        // Assert
        Assert.NotNull(result);
        Assert.Contains("Description", result.TextColumnNames);
        Assert.Contains("Price", result.NumericColumnNames);
        Assert.Equal("Label", result.LabelColumnName);
    }

    [Fact]
    public void BuildColumnInformation_LabelColumnExcluded_NotInAnyFeatureList()
    {
        // Arrange
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new TextBinaryData { Content = "text", Label = true },
        });

        // Act
        var result = AutoMLRunner.BuildColumnInformation(data, "Label");

        // Assert
        Assert.NotNull(result);
        Assert.DoesNotContain("Label", result.TextColumnNames);
        Assert.DoesNotContain("Label", result.NumericColumnNames);
        Assert.DoesNotContain("Label", result.IgnoredColumnNames);
    }

    [Fact]
    public void BuildColumnInformation_MultipleTextColumns_AllClassified()
    {
        // Arrange: multiple text columns
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new MultiTextData { Title = "Great", Body = "Detailed review text", Label = true },
            new MultiTextData { Title = "Bad", Body = "Short negative review", Label = false },
        });

        // Act
        var result = AutoMLRunner.BuildColumnInformation(data, "Label");

        // Assert
        Assert.NotNull(result);
        Assert.Contains("Title", result.TextColumnNames);
        Assert.Contains("Body", result.TextColumnNames);
        Assert.Equal(2, result.TextColumnNames.Count);
    }

    [Fact]
    public void BuildColumnInformation_CaseInsensitiveLabelMatch()
    {
        // Arrange
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new TextBinaryData { Content = "text", Label = true },
        });

        // Act — label column name case doesn't match property name exactly
        var result = AutoMLRunner.BuildColumnInformation(data, "label");

        // Assert: Label should still be excluded from features
        Assert.NotNull(result);
        Assert.Single(result.TextColumnNames);
        Assert.Contains("Content", result.TextColumnNames);
    }

    #endregion

    #region ColumnOverride Tests

    [Fact]
    public void BuildColumnInformation_WithTextOverride_ForcesTextClassification()
    {
        // Arrange: numeric-only dataset, but override Feature1 as text
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new NumericData { Feature1 = 1.0f, Feature2 = 2.0f, Label = true },
        });
        // Note: ML.NET loads float as NumberDataViewType, but override should still add to correct list
        var overrides = new Dictionary<string, string> { ["Feature1"] = "text" };

        // Act
        var result = AutoMLRunner.BuildColumnInformation(data, "Label", columnOverrides: overrides);

        // Assert: non-null because override was applied
        Assert.NotNull(result);
        Assert.Contains("Feature1", result.TextColumnNames);
        Assert.Contains("Feature2", result.NumericColumnNames);
    }

    [Fact]
    public void BuildColumnInformation_WithIgnoreOverride_ExcludesColumn()
    {
        // Arrange: text dataset, but override Content as ignore
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new TextBinaryData { Content = "some text", Label = true },
        });
        var overrides = new Dictionary<string, string> { ["Content"] = "ignore" };

        // Act
        var result = AutoMLRunner.BuildColumnInformation(data, "Label", columnOverrides: overrides);

        // Assert
        Assert.NotNull(result);
        Assert.DoesNotContain("Content", result.TextColumnNames);
        Assert.Contains("Content", result.IgnoredColumnNames);
    }

    [Fact]
    public void BuildColumnInformation_WithCategoricalOverride_ClassifiesAsCategorical()
    {
        // Arrange: text dataset, but override Content as categorical
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new TextBinaryData { Content = "category_A", Label = true },
        });
        var overrides = new Dictionary<string, string> { ["Content"] = "categorical" };

        // Act
        var result = AutoMLRunner.BuildColumnInformation(data, "Label", columnOverrides: overrides);

        // Assert
        Assert.NotNull(result);
        Assert.DoesNotContain("Content", result.TextColumnNames);
        Assert.Contains("Content", result.CategoricalColumnNames);
    }

    [Fact]
    public void BuildColumnInformation_WithNumericOverride_ClassifiesAsNumeric()
    {
        // Arrange: text dataset, but override Description as numeric
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new MixedData { Description = "text", Price = 10.0f, Label = true },
        });
        var overrides = new Dictionary<string, string> { ["Description"] = "numeric" };

        // Act
        var result = AutoMLRunner.BuildColumnInformation(data, "Label", columnOverrides: overrides);

        // Assert: Description forced to numeric — no text columns remain,
        // so ColumnInformation is null (AutoML default behavior handles numeric fine)
        Assert.Null(result);
    }

    [Fact]
    public void BuildColumnInformation_WithMixedOverrides_AllClassifiedCorrectly()
    {
        // Arrange: dataset with text + numeric, apply mixed overrides
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new MultiTextData { Title = "title", Body = "body text", Label = true },
        });
        var overrides = new Dictionary<string, string>
        {
            ["Title"] = "categorical",  // force text → categorical
            // Body stays as text (no override)
        };

        // Act
        var result = AutoMLRunner.BuildColumnInformation(data, "Label", columnOverrides: overrides);

        // Assert
        Assert.NotNull(result);
        Assert.Contains("Title", result.CategoricalColumnNames);
        Assert.Contains("Body", result.TextColumnNames);
        Assert.DoesNotContain("Title", result.TextColumnNames);
    }

    [Fact]
    public void BuildColumnInformation_NoOverrides_BehavesAsOriginal()
    {
        // Arrange
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new NumericData { Feature1 = 1.0f, Feature2 = 2.0f, Label = true },
        });

        // Act: pass empty overrides
        var result = AutoMLRunner.BuildColumnInformation(data, "Label", columnOverrides: new Dictionary<string, string>());

        // Assert: same as no overrides — returns null for numeric-only
        Assert.Null(result);
    }

    #endregion

    #region EnsureCalibratedModel Tests (D15 — BUG-24 follow-through)

    [Fact]
    public void EnsureCalibratedModel_AlreadyCalibrated_ReturnsSameModel()
    {
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new NumericData { Feature1 = 1f, Feature2 = 1f, Label = true },
            new NumericData { Feature1 = 5f, Feature2 = 5f, Label = false },
        });
        var featurized = _mlContext.Transforms.Concatenate("Features", "Feature1", "Feature2").Fit(data).Transform(data);
        var model = _mlContext.BinaryClassification.Trainers
            .SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features").Fit(featurized);
        var predictions = model.Transform(featurized);

        var result = AutoMLRunner.EnsureCalibratedModel(_mlContext, model, predictions, "Label", hasProbability: true);

        Assert.Same(model, result);
    }

    [Fact]
    public void EnsureCalibratedModel_UncalibratedTrainer_AppendsPlattCalibratorRestoringProbability()
    {
        // D15: reproduces the KAMP SEQ089 live finding — FastForestBinary (AutoML's chosen "best
        // trainer") emits Score but no Probability. Uncalibrated, HoneAI's ConfidenceOf reads no
        // probabilities/score signal and falls back to 0 for every row (100% escalation, observed
        // live via honeai-sim). The saved model must carry a calibrated Probability so downstream
        // predict/serve consumers get a real confidence signal.
        var data = _mlContext.Data.LoadFromEnumerable(Enumerable.Range(0, 40).Select(i => new NumericData
        {
            Feature1 = i < 20 ? 1f + i * 0.05f : 5f + i * 0.05f,
            Feature2 = i < 20 ? 1f + i * 0.05f : 5f + i * 0.05f,
            Label = i < 20,
        }));
        var featurized = _mlContext.Transforms.Concatenate("Features", "Feature1", "Feature2").Fit(data).Transform(data);
        var model = _mlContext.BinaryClassification.Trainers
            .FastForest(labelColumnName: "Label", featureColumnName: "Features").Fit(featurized);
        var predictions = model.Transform(featurized);

        // Precondition: FastForestBinary alone genuinely produces no Probability column.
        Assert.Null(predictions.Schema.GetColumnOrNull("Probability"));

        var calibrated = AutoMLRunner.EnsureCalibratedModel(_mlContext, model, predictions, "Label", hasProbability: false);
        var calibratedPredictions = calibrated.Transform(featurized);

        var probCol = calibratedPredictions.Schema.GetColumnOrNull("Probability");
        Assert.NotNull(probCol);

        using var cursor = calibratedPredictions.GetRowCursor(calibratedPredictions.Schema);
        var getter = cursor.GetGetter<float>(probCol.Value);
        var sawAny = false;
        while (cursor.MoveNext())
        {
            float p = default;
            getter(ref p);
            Assert.InRange(p, 0f, 1f);
            sawAny = true;
        }
        Assert.True(sawAny);
    }

    #endregion

    #region ComputeConformalIntervals Tests (② regression wave — prediction interval)

    [Fact]
    public void ComputeConformalIntervals_KnownResiduals_ProducesFiniteSampleQuantiles()
    {
        // Residuals |Label - Score| = {1,2,...,10} (n=10): Label=0, Score={1..10}.
        // Split-conformal rank = ceil((n+1)*level): level .80 -> ceil(8.8)=9 -> 9th smallest = 9;
        // level .90 -> ceil(9.9)=10 -> 10; level .95 -> ceil(10.45)=11 > n -> clamp to max = 10.
        var data = _mlContext.Data.LoadFromEnumerable(
            Enumerable.Range(1, 10).Select(i => new RegressionScored { Label = 0f, Score = i }));

        var intervals = AutoMLRunner.ComputeConformalIntervals(data, "Label");

        Assert.Equal(9.0, intervals["interval_half_width_80"], 6);
        Assert.Equal(10.0, intervals["interval_half_width_90"], 6);
        Assert.Equal(10.0, intervals["interval_half_width_95"], 6); // clamped (level too high for n=10)
        // residual_std = sqrt((1+4+...+100)/10) = sqrt(38.5)
        Assert.Equal(Math.Sqrt(38.5), intervals["residual_std"], 6);
    }

    [Fact]
    public void ComputeConformalIntervals_EmpiricalCoverage_MeetsTargetLevel()
    {
        // The conformal guarantee: the fraction of holdout residuals within ±q must be ≥ the level.
        var data = _mlContext.Data.LoadFromEnumerable(
            Enumerable.Range(0, 200).Select(i => new RegressionScored { Label = 0f, Score = (i % 100) * 0.1f }));

        var intervals = AutoMLRunner.ComputeConformalIntervals(data, "Label");
        var residuals = Enumerable.Range(0, 200).Select(i => Math.Abs((i % 100) * 0.1)).ToList();

        foreach (var (pct, level) in new[] { (80, 0.80), (90, 0.90) })
        {
            double q = intervals[$"interval_half_width_{pct}"];
            double coverage = residuals.Count(r => r <= q + 1e-9) / (double)residuals.Count;
            Assert.True(coverage >= level, $"coverage {coverage} < level {level} for {pct}%");
        }
        // Higher confidence => wider (never narrower) band.
        Assert.True(intervals["interval_half_width_95"] >= intervals["interval_half_width_90"]);
        Assert.True(intervals["interval_half_width_90"] >= intervals["interval_half_width_80"]);
    }

    [Fact]
    public void ComputeConformalIntervals_MissingScoreColumn_ReturnsEmpty()
    {
        // Graceful: a data view with no Score column yields no band (never throws).
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new NumericData { Feature1 = 1f, Feature2 = 1f, Label = true },
        });

        var intervals = AutoMLRunner.ComputeConformalIntervals(data, "Label");

        Assert.Empty(intervals);
    }

    #endregion

    #region ComputeNormalizedConformal Tests (② regression wave — heteroscedastic per-row band)

    // Residual magnitude grows linearly with X: |Label - Score| = 0.5*X. A per-row σ(X) model should
    // therefore recover width ∝ X, so the widest bands land on the largest-error rows.
    private static IEnumerable<HeteroScored> HeteroData(int n) =>
        Enumerable.Range(0, n).Select(i =>
        {
            float x = (i % 50) + 1;                                   // X in [1,50]
            float signedResid = 0.5f * x * ((i % 2 == 0) ? 1 : -1);  // |Label - Score| = 0.5*X
            return new HeteroScored { X = x, Label = 0f, Score = signedResid };
        });

    // width = q * (max(σ_raw, 0) + β) per row, paired with the true |residual|.
    private List<(double width, double absResid)> PerRowWidths(
        NormalizedConformalResult result, IEnumerable<HeteroScored> eval, int pct = 90)
    {
        var rows = eval.ToList();
        var view = _mlContext.Data.LoadFromEnumerable(rows);
        var scored = result.AuxModel.Transform(view);
        var col = scored.Schema["Score"];
        double q = result.Metrics[$"norm_interval_q_{pct}"];
        double beta = result.Metrics["interval_beta"];
        var widths = new List<(double, double)>();
        using var cursor = scored.GetRowCursor(new[] { col });
        var getter = cursor.GetGetter<float>(col);
        int i = 0;
        float sigmaRaw = 0;
        while (cursor.MoveNext())
        {
            getter(ref sigmaRaw);
            double width = q * (Math.Max(sigmaRaw, 0.0) + beta);
            widths.Add((width, Math.Abs(rows[i].Label - rows[i].Score)));
            i++;
        }
        return widths;
    }

    [Fact]
    public void ComputeNormalizedConformal_HeteroscedasticData_WidthRanksLargeErrorRows()
    {
        // The M-13 defect: a constant-width band cannot rank the review-worthy (large-error) rows —
        // its recall equals random. The per-row band must rank them well above random.
        var data = _mlContext.Data.LoadFromEnumerable(HeteroData(200));

        var result = AutoMLRunner.ComputeNormalizedConformal(_mlContext, data, "Label", new[] { "X" });

        Assert.NotNull(result);
        Assert.True(result!.Metrics.ContainsKey("norm_interval_q_90"));
        Assert.True(result.Metrics.ContainsKey("interval_beta"));

        var eval = PerRowWidths(result, HeteroData(200));
        int n = eval.Count, k = Math.Max(1, n / 10);
        var worst = eval.Select((e, i) => (e.absResid, i)).OrderByDescending(t => t.absResid)
            .Take(k).Select(t => t.i).ToHashSet();
        var widest = eval.Select((e, i) => (e.width, i)).OrderByDescending(t => t.width)
            .Take(k).Select(t => t.i).ToHashSet();
        double recall = widest.Count(worst.Contains) / (double)worst.Count;
        Assert.True(recall >= 0.5, $"per-row width recall {recall:F2} of large-error rows should beat random (~0.1)");
    }

    [Fact]
    public void ComputeNormalizedConformal_EmpiricalCoverage_MeetsTargetLevel()
    {
        // Normalized-conformal guarantee: fraction of rows with |resid| ≤ q·σ(x) must be ≥ the level.
        var data = _mlContext.Data.LoadFromEnumerable(HeteroData(200));

        var result = AutoMLRunner.ComputeNormalizedConformal(_mlContext, data, "Label", new[] { "X" });

        Assert.NotNull(result);
        foreach (var pct in new[] { 80, 90 })
        {
            var eval = PerRowWidths(result!, HeteroData(200), pct);
            double coverage = eval.Count(e => e.absResid <= e.width + 1e-6) / (double)eval.Count;
            Assert.True(coverage >= pct / 100.0 - 0.05, $"coverage {coverage:F2} < {pct}% level");
        }
    }

    [Fact]
    public void ComputeNormalizedConformal_HomoscedasticData_WidthNearlyConstant()
    {
        // Backward-safe: when the error does not depend on the features, σ(x) degenerates toward a
        // constant and the per-row band collapses to (near) constant width.
        var homo = Enumerable.Range(0, 200).Select(i =>
            new HeteroScored { X = (i % 50) + 1, Label = 0f, Score = ((i % 2 == 0) ? 3f : -3f) }); // |resid|=3 always
        var data = _mlContext.Data.LoadFromEnumerable(homo);

        var result = AutoMLRunner.ComputeNormalizedConformal(_mlContext, data, "Label", new[] { "X" });

        Assert.NotNull(result);
        var widths = PerRowWidths(result!, homo).Select(e => e.width).ToList();
        double mean = widths.Average();
        double spread = (widths.Max() - widths.Min()) / mean;
        Assert.True(spread < 0.20, $"homoscedastic width spread {spread:F2} should be small (near-constant)");
    }

    [Fact]
    public void ComputeNormalizedConformal_ThroughRealModelTransform_ProducesAuxModel()
    {
        // Mirrors the production path: a fitted regression model's Transform output (raw feature
        // columns + "Features" + "Score") is what RunRegressionAsync passes in. Guards the wiring
        // assumption that the raw numeric columns survive the transform for the σ-model to read.
        var rows = Enumerable.Range(0, 200).Select(i =>
        {
            float f1 = (i % 50) + 1;
            return new RealReg { F1 = f1, F2 = i % 7, MX = 2f * f1 + 0.5f * f1 * ((i % 2 == 0) ? 1 : -1) };
        }).ToList();
        var data = _mlContext.Data.LoadFromEnumerable(rows);
        var split = _mlContext.Data.TrainTestSplit(data, 0.3, seed: 1);
        var pipeline = _mlContext.Transforms.Concatenate("Features", "F1", "F2")
            .Append(_mlContext.Regression.Trainers.Sdca(labelColumnName: "MX", featureColumnName: "Features"));
        var model = pipeline.Fit(split.TrainSet);
        var scored = model.Transform(split.TestSet);

        var result = AutoMLRunner.ComputeNormalizedConformal(_mlContext, scored, "MX", new[] { "F1", "F2" });

        Assert.NotNull(result);
        Assert.True(result!.Metrics.ContainsKey("norm_interval_q_90"));
    }

    [Fact]
    public void ComputeNormalizedConformal_MissingScoreOrNoNumericFeature_ReturnsNull()
    {
        // Graceful: no Score column, or no usable numeric feature → null (caller uses constant-width).
        var noScore = _mlContext.Data.LoadFromEnumerable(new[] { new NumericData { Feature1 = 1f, Label = true } });
        Assert.Null(AutoMLRunner.ComputeNormalizedConformal(_mlContext, noScore, "Label", new[] { "Feature1" }));

        var noFeature = _mlContext.Data.LoadFromEnumerable(HeteroData(50));
        Assert.Null(AutoMLRunner.ComputeNormalizedConformal(_mlContext, noFeature, "Label", new[] { "Nonexistent" }));
    }

    #endregion

    #region Test Data Classes

    private class RegressionScored
    {
        public float Label { get; set; }
        public float Score { get; set; }
    }

    private class HeteroScored
    {
        public float X { get; set; }
        public float Label { get; set; }
        public float Score { get; set; }
    }

    private class RealReg
    {
        public float F1 { get; set; }
        public float F2 { get; set; }
        public float MX { get; set; }
    }

    private class TextBinaryData
    {
        public string Content { get; set; } = "";
        public bool Label { get; set; }
    }

    private class NumericData
    {
        public float Feature1 { get; set; }
        public float Feature2 { get; set; }
        public bool Label { get; set; }
    }

    private class MixedData
    {
        public string Description { get; set; } = "";
        public float Price { get; set; }
        public bool Label { get; set; }
    }

    private class MultiTextData
    {
        public string Title { get; set; } = "";
        public string Body { get; set; } = "";
        public bool Label { get; set; }
    }

    #endregion
}
