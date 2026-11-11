using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using MLoop.Core.Models;
using MLoop.Core.Storage;

namespace MLoop.Core.Tests.Models;

/// <summary>
/// What a table anomaly model's score and answer mean: the raw squared distance from the components' rebuild, "anomalous" above
/// the point a chosen share of the training rows score above. ML.NET's own score is that distance divided by the row's own length
/// and its answer is a fixed 0.5 — a number from no data.
/// </summary>
public class AnomalyScoringTests
{
    private sealed class Row
    {
        [VectorType(4)]
        public float[] Features { get; set; } = [];
    }

    // Three columns that move together plus one of noise; the last `anomalies` rows break the shared movement.
    private static List<Row> Rows(int count, int anomalies, int seed)
    {
        var random = new Random(seed);
        double Gauss() => Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble());
        var rows = new List<Row>();
        for (var i = 0; i < count; i++)
        {
            var broken = i >= count - anomalies;
            var z = Gauss();
            double Column() => broken ? 3 * Gauss() : z + 0.05 * Gauss();
            rows.Add(new Row { Features = [(float)Column(), (float)Column(), (float)Column(), (float)Gauss()] });
        }
        return rows;
    }

    private static (MLContext Ml, IDataView Data, AnomalyPredictionTransformer<PcaModelParameters> Pca) Fit(List<Row> rows, int rank = 2)
    {
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(rows);
        var pca = ml.AnomalyDetection.Trainers.RandomizedPca("Features", rank: rank, oversampling: 20).Fit(data);
        return (ml, data, pca);
    }

    [Fact]
    public void TheScore_IsTheRebuildDistance_NotTheShareOfTheRowsOwnLength()
    {
        var rows = Rows(400, 20, seed: 3);
        var (ml, data, pca) = Fit(rows);
        var scoring = AnomalyScoring.Fit(data, pca.Model, 0.05);
        var scored = scoring.Apply(ml, pca).Transform(data);

        var own = ml.Data.CreateEnumerable<Scored>(pca.Transform(data), reuseRowObject: false).ToList();
        var raw = ml.Data.CreateEnumerable<Scored>(scored, reuseRowObject: false).ToList();
        VBuffer<float>[] axes = default!;
        pca.Model.GetEigenVectors(ref axes, out _);
        VBuffer<float> meanBuffer = default;
        pca.Model.GetMean(ref meanBuffer);
        var mean = meanBuffer.DenseValues().ToArray();

        for (var i = 0; i < rows.Count; i++)
        {
            // Independent of the code under test: the squared distance by the long way, from the axes and the mean.
            var centred = rows[i].Features.Select((v, j) => (double)v - mean[j]).ToArray();
            var carried = axes.Sum(a => Math.Pow(a.DenseValues().Select((c, j) => c * centred[j]).Sum(), 2));
            var distance = centred.Sum(v => v * v) - carried;
            Assert.Equal(distance, raw[i].Score, 3);

            // ML.NET's score is that distance relative to the row's own length: the two differ by a per-row factor.
            Assert.True(own[i].Score >= 0 && own[i].Score <= 1.0001f);
        }
        // The two orders differ — which is the whole reason the raw one is written.
        Assert.NotEqual(
            own.Select((s, i) => (s.Score, i)).OrderByDescending(p => p.Score).First().i,
            raw.Select((s, i) => (s.Score, i)).OrderByDescending(p => p.Score).First().i);
    }

    private sealed class Scored
    {
        public float Score { get; set; }
        public bool PredictedLabel { get; set; }
    }

    [Fact]
    public void TheThreshold_IsPlacedSoThatTheChosenShareOfTheTrainingRowsScoreAboveIt()
    {
        var rows = Rows(1000, 50, seed: 4);
        var (ml, data, pca) = Fit(rows);

        foreach (var contamination in new[] { 0.01, 0.05, 0.2 })
        {
            var scoring = AnomalyScoring.Fit(data, pca.Model, contamination);
            var flagged = ml.Data.CreateEnumerable<Scored>(scoring.Apply(ml, pca).Transform(data), reuseRowObject: false).Count(s => s.PredictedLabel);
            Assert.InRange(flagged, (int)(1000 * contamination) - 1, (int)(1000 * contamination) + 1);
            Assert.Equal(1000, scoring.Rows);
            Assert.Equal(contamination, scoring.Contamination);
        }
    }

    [Fact]
    public void TheAnomalies_AreAmongTheFlaggedRows_AndTheNormalRowsMostlyAreNot()
    {
        var rows = Rows(1000, 50, seed: 5);
        var (ml, data, pca) = Fit(rows);
        var scoring = AnomalyScoring.Fit(data, pca.Model, 0.05);
        var answers = ml.Data.CreateEnumerable<Scored>(scoring.Apply(ml, pca).Transform(data), reuseRowObject: false).ToList();

        var caught = answers.Skip(950).Count(a => a.PredictedLabel);
        var falseAlarms = answers.Take(950).Count(a => a.PredictedLabel);
        Assert.True(caught >= 40, $"{caught} of the 50 rows that break the shared movement were flagged");
        Assert.True(falseAlarms <= 20, $"{falseAlarms} of the 950 ordinary rows were flagged");
    }

    [Fact]
    public void ARowAtTheThreshold_IsNotAbove_AndTiesCannotPushTheShareOver()
    {
        // Ten equal scores: the share above the 90th percentile is none, not all ten.
        Assert.Equal(1.0, AnomalyScoring.QuantileAbove(Enumerable.Repeat(1.0, 10).ToArray(), 0.1));
        // 1..100: the top 5 lie above 95.
        Assert.Equal(95.0, AnomalyScoring.QuantileAbove(Enumerable.Range(1, 100).Select(i => (double)i).ToArray(), 0.05));
        Assert.Equal(10.0, AnomalyScoring.QuantileAbove([10.0], 0.25));
    }

    [Theory]
    [InlineData(0.05, true)]
    [InlineData(0.49, true)]
    [InlineData(0.0, false)]
    [InlineData(0.5, false)]
    [InlineData(-0.1, false)]
    [InlineData(double.NaN, false)]
    public void TheContamination_IsAShareAboveZeroAndBelowOneHalf(double value, bool valid)
        => Assert.Equal(valid, AnomalyScoring.IsValidContamination(value));

    [Fact]
    public void AModelThatIsNotAnAnomalyModel_IsAnsweredUnchanged()
    {
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(Rows(50, 0, seed: 6));
        var other = ml.Transforms.NormalizeMeanVariance("Features").Fit(data);
        Assert.Same(other, new AnomalyScoring(1, 0.05, 50).Apply(ml, other));
    }

    [Fact]
    public void TheFile_RoundTripsBesideTheModel_AndNoFileIsNoLayer()
    {
        var dir = Path.Combine(Path.GetTempPath(), "mloop-anomaly-scoring-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var model = Path.Combine(dir, ExperimentLayout.ModelFileName);
            Assert.Null(AnomalyScoring.ReadBeside(model));
            new AnomalyScoring(2.5, 0.1, 300).WriteBeside(model);
            var read = AnomalyScoring.ReadBeside(model);
            Assert.Equal(new AnomalyScoring(2.5, 0.1, 300), read);
            Assert.True(File.Exists(Path.Combine(dir, ExperimentLayout.AnomalyScoringFileName)));
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    private sealed class LabeledRow
    {
        [VectorType(4)]
        public float[] Features { get; set; } = [];
        public float Label { get; set; }
    }

    [Fact]
    public void TheLayeredOutput_StillGoesThroughMlNetsEvaluation_AndCarriesItsThreshold()
    {
        var plain = Rows(1000, 50, seed: 8);
        var labeled = plain.Select((r, i) => new LabeledRow { Features = r.Features, Label = i >= 950 ? 1f : 0f }).ToList();
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(labeled);
        var pca = ml.AnomalyDetection.Trainers.RandomizedPca("Features", rank: 2, oversampling: 20).Fit(data);
        var scoring = AnomalyScoring.Fit(data, pca.Model, 0.05);
        var scored = scoring.Apply(ml, pca).Transform(data);

        var metrics = ml.AnomalyDetection.Evaluate(scored, "Label");
        Assert.True(metrics.AreaUnderRocCurve > 0.9, $"area under the curve {metrics.AreaUnderRocCurve:F3}");
        Assert.Equal(scoring.Threshold, AnomalyScoring.ThresholdOf(scored.Schema)!.Value, 3);
        Assert.Null(AnomalyScoring.ThresholdOf(pca.Transform(data).Schema));
    }

    [Fact]
    public void ASavedAndReloadedModel_IsFoundAndScoredLikeTheOneThatWasFitted()
    {
        var rows = Rows(500, 25, seed: 9);
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(rows);
        var featurizer = ml.Transforms.NormalizeMeanVariance("Features", fixZero: false).Fit(data);
        var pca = ml.AnomalyDetection.Trainers.RandomizedPca("Features", rank: 2, oversampling: 20).Fit(featurizer.Transform(data));
        var fitted = featurizer.Append(pca);

        var path = Path.Combine(Path.GetTempPath(), $"anomaly-scoring-{Guid.NewGuid():N}.zip");
        try
        {
            ml.Model.Save(fitted, data.Schema, path);
            var loaded = ml.Model.Load(path, out _);

            Assert.NotNull(AnomalyScoring.PcaOf(loaded));
            var scoring = AnomalyScoring.Fit(featurizer.Transform(data), pca.Model, 0.05);
            var a = ml.Data.CreateEnumerable<Scored>(scoring.Apply(ml, fitted).Transform(data), reuseRowObject: false).ToList();
            var b = ml.Data.CreateEnumerable<Scored>(scoring.Apply(ml, loaded).Transform(data), reuseRowObject: false).ToList();
            var worst = a.Zip(b, (x, y) => Math.Abs(x.Score - y.Score)).Max();
            var scale = a.Max(x => x.Score);
            Assert.True(worst <= 1e-4 * scale, $"largest difference {worst} on scores up to {scale}");
            Assert.Equal(a.Count(x => x.PredictedLabel), b.Count(x => x.PredictedLabel));
            Assert.InRange(b.Count(x => x.PredictedLabel), 20, 30);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ARowWithAMissingValue_HasNoScore_AndIsNotCalledNormal()
    {
        var rows = Rows(200, 10, seed: 7);
        var (ml, data, pca) = Fit(rows);
        var scoring = AnomalyScoring.Fit(data, pca.Model, 0.05);
        var withGap = ml.Data.LoadFromEnumerable(new[] { new Row { Features = [float.NaN, 0, 0, 0] } });
        var answer = ml.Data.CreateEnumerable<Scored>(scoring.Apply(ml, pca).Transform(withGap), reuseRowObject: false).Single();
        Assert.True(float.IsNaN(answer.Score));
    }
}
