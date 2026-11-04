using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Search.Candidates;
using MLoop.Search.Validation;

namespace MLoop.Search.Tests;

/// <summary>
/// <see cref="SearchSample"/> — searching a large table on a sample of its rows: the arithmetic that sizes the sample, the sample
/// itself, and the search that probes a table from <see cref="SearchSample.ProbeRows"/> rows and leaves a smaller one untouched.
/// </summary>
public sealed class SearchSampleTests
{
    // ---- the arithmetic ----

    [Fact]
    public void ATableBelowTheProbeThreshold_IsSearchedWhole_WhateverTheProbeSaid()
        => Assert.Equal(SearchSample.ProbeRows - 1, SearchSample.RowsToSearch(SearchSample.ProbeRows - 1, 60, probeSeconds: 1_000, SearchSample.ProbeSample));

    [Fact]
    public void AFirstTrialWithinATenthOfTheBudget_IsRunOnAllTheRows()
    {
        // 10,000 rows fitted twice in 0.2 s → 1e-5 s a fitted row; a first trial on 200,000 rows fits 180,000 of them → 1.8 s, inside 6 s.
        Assert.Equal(200_000, SearchSample.RowsToSearch(200_000, 60, probeSeconds: 0.2, SearchSample.ProbeSample));
    }

    [Fact]
    public void AFirstTrialOverATenthOfTheBudget_IsRunOnTheLargestStepOfSampleWhoseFirstTrialIsNot()
    {
        // 23 s for the cheapest first trial of 80,000 rows (measured): a probe of 4.6 s for 10,000 rows.
        var rows = SearchSample.RowsToSearch(100_000, budgetSeconds: 60, probeSeconds: 4.6, SearchSample.ProbeSample);
        Assert.Equal(20_000, rows);   // 28,985 rows would fit; the steps are 10,000 × 2^k
        var rate = SearchSample.SecondsPerFittedRow(4.6, SearchSample.ProbeSample);
        Assert.True(rate * SearchSample.FittedRows(rows) <= SearchSample.FirstTrialShare * 60 + 1e-9);
        Assert.True(rate * SearchSample.FittedRows(rows * 2) > SearchSample.FirstTrialShare * 60);
    }

    [Fact]
    public void TheSampleSize_DoesNotMoveWithSmallDifferencesInTheProbeTime()
        => Assert.Equal(
            SearchSample.RowsToSearch(1_000_000, 60, 4.0, SearchSample.ProbeSample),
            SearchSample.RowsToSearch(1_000_000, 60, 4.3, SearchSample.ProbeSample));

    [Fact]
    public void ATableTooBigForAnySampleTheBudgetAllows_IsSearchedOnTheProbeSample()
        => Assert.Equal(SearchSample.ProbeSample, SearchSample.RowsToSearch(1_000_000, budgetSeconds: 60, probeSeconds: 30, SearchSample.ProbeSample));

    [Fact]
    public void ABiggerBudget_BuysABiggerSample()
    {
        var small = SearchSample.RowsToSearch(1_000_000, 60, 3, SearchSample.ProbeSample);
        var large = SearchSample.RowsToSearch(1_000_000, 600, 3, SearchSample.ProbeSample);
        Assert.True(large > small, $"{large} vs {small}");
    }

    [Fact]
    public void AProbeThatTookNoTime_SearchesAllTheRows()
        => Assert.Equal(500_000, SearchSample.RowsToSearch(500_000, 60, probeSeconds: 0, SearchSample.ProbeSample));

    [Theory]
    [InlineData(1_000_000, 10_000, 1.0, 60.0, false)]   // 100x the rows: 100 s, over a 30 s share
    [InlineData(1_000_000, 10_000, 0.1, 60.0, true)]    // 10 s, inside it
    [InlineData(1_000_000, 10_000, 0.5, 60.0, false)]   // 50 s, over it
    [InlineData(1_000_000, 1_000_000, 500.0, 60.0, true)] // not sampled: nothing to scale
    public void ARefitOnAllTheRows_IsMadeWhenItsPredictionFitsHalfTheBudget(int rows, int sampleRows, double sampleSeconds, double budget, bool expected)
        => Assert.Equal(expected, SearchSample.RefitsOnAllRows(rows, sampleRows, sampleSeconds, budget));

    // ---- the sample ----

    [Fact]
    public void AStratifiedSample_KeepsEachClassesShare_AndTheSizeAsked()
    {
        var keys = Enumerable.Range(0, 100_000).Select(i => (i % 10) switch { < 6 => "a", < 9 => "b", _ => "c" }).ToArray();   // 60 / 30 / 10 %
        var keep = SearchSample.Stratified(keys, stratify: true, 10_000, seed: 3);
        Assert.Equal(10_000, keep.Count(k => k));
        Assert.Equal(6_000, keys.Where((_, i) => keep[i]).Count(k => k == "a"));
        Assert.Equal(3_000, keys.Where((_, i) => keep[i]).Count(k => k == "b"));
        Assert.Equal(1_000, keys.Where((_, i) => keep[i]).Count(k => k == "c"));
    }

    [Fact]
    public void ASample_IsTheSameForTheSameSeed_AndDiffersForAnother()
    {
        var keys = Enumerable.Range(0, 5_000).Select(i => (i % 3).ToString()).ToArray();
        Assert.Equal(SearchSample.Stratified(keys, true, 500, 1), SearchSample.Stratified(keys, true, 500, 1));
        Assert.NotEqual(SearchSample.Stratified(keys, true, 500, 1), SearchSample.Stratified(keys, true, 500, 2));
    }

    [Fact]
    public void ASampleOfARegression_IsDrawnAtRandom_NotFromTheFront()
    {
        var keys = Enumerable.Range(0, 5_000).Select(i => i.ToString()).ToArray();   // a label that never repeats
        var keep = SearchSample.Stratified(keys, stratify: false, 500, seed: 1);
        Assert.Equal(500, keep.Count(k => k));
        Assert.True(keep.Skip(2_500).Count(k => k) > 150, "the sample reaches the back half of the rows");
        Assert.True(keep.Take(2_500).Count(k => k) > 150, "and the front half");
    }

    [Fact]
    public void ASampleAsBigAsTheTable_KeepsEveryRow()
        => Assert.All(SearchSample.Stratified(["a", "b", "a"], true, 3, 1), Assert.True);

    [Fact]
    public void TheSampledRows_AreTheFlaggedOnes_InTheirOrder_WithoutExtraColumns()
    {
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 20).Select(i => new Point { Id = i, Label = i % 2 == 0 }));
        var keep = Enumerable.Range(0, 20).Select(i => i % 3 == 0).ToArray();
        var view = SearchSample.Rows(ml, data, keep);
        Assert.Equal(data.Schema.Count, view.Schema.Count);
        Assert.Equal([0, 3, 6, 9, 12, 15, 18], view.GetColumn<int>("Id").ToArray());
        // a fold's view can be made on it: the holdout column of the sample's own is not in the way
        var holdout = new HoldoutView(view, Enumerable.Repeat(0, 7).ToArray(), 0);
        Assert.Equal(data.Schema.Count + 1, holdout.Schema.Count);
    }

    private sealed class Point
    {
        public int Id { get; set; }
        public bool Label { get; set; }
    }

    // ---- the search ----

    private sealed class Row
    {
        public bool Label { get; set; }
        public float X { get; set; }
        public float Z { get; set; }
    }

    private sealed record Auc(string Name = "auc", bool HigherIsBetter = true) : IMetricPolicy;

    private static (IDataView Data, ColumnRoles Roles) Table(MLContext ml, int n)
    {
        var random = new Random(11);
        var rows = Enumerable.Range(0, n).Select(_ =>
        {
            var x = (float)random.NextDouble();
            var z = (float)random.NextDouble();
            return new Row { Label = x + 0.5 * z + random.NextDouble() * 0.4 > 0.9, X = x, Z = z };
        }).ToList();
        return (ml.Data.LoadFromEnumerable(rows), new ColumnRoles("Label", [new("X", ColumnRole.Numeric), new("Z", ColumnRole.Numeric)]));
    }

    private static SearchRequest Request(MLContext ml, TimeSpan budget, bool sample, int? maxTrials = null) => new(
        CandidateCatalog.Binary(name => name is "FastTree" or "LbfgsLogisticRegression"),
        new Auc(),
        scored => new Dictionary<string, double> { ["auc"] = ml.BinaryClassification.EvaluateNonCalibrated(scored).AreaUnderRocCurve },
        budget,
        new SearchSettings(Seed: 3),
        MaxTrials: maxTrials,
        SampleLargeTables: sample);

    [Fact]
    public void ATableBelowTheThreshold_IsSearchedExactlyAsWithoutProbing()
    {
        var ml = new MLContext(seed: 1);
        var (data, roles) = Table(ml, 3_000);
        var probed = new TrialRunner(Request(ml, TimeSpan.FromSeconds(60), sample: true, maxTrials: 5)).Run(data, roles);
        var plain = new TrialRunner(Request(ml, TimeSpan.FromSeconds(60), sample: false, maxTrials: 5)).Run(data, roles);
        Assert.Null(probed.Sampling);
        Assert.Equal(plain.Trials.Select(t => t.Trainer), probed.Trials.Select(t => t.Trainer));
        Assert.Equal(plain.Trials.Select(t => t.Metrics!["auc"]), probed.Trials.Select(t => t.Metrics!["auc"]));
        Assert.Equal(plain.Trials.Select(t => string.Join(",", t.Parameters!.Select(p => p.Value))), probed.Trials.Select(t => string.Join(",", t.Parameters!.Select(p => p.Value))));
    }

    [Fact]
    public void ATableTooBigForItsBudget_IsSearchedOnASample_AndTheModelStillScoresEveryRow()
    {
        var ml = new MLContext(seed: 1);
        var (data, roles) = Table(ml, SearchSample.ProbeRows + 5_000);
        // A second is a tenth of a second for the first trial: no trial on a hundred thousand rows fits it, however fast the machine.
        var result = new TrialRunner(Request(ml, TimeSpan.FromSeconds(1), sample: true)).Run(data, roles);

        var sampling = Assert.IsType<SamplingOutcome>(result.Sampling);
        Assert.True(sampling.Sampled);
        Assert.Equal(SearchSample.ProbeRows + 5_000, sampling.Rows);
        Assert.InRange(sampling.SearchRows, SearchSample.ProbeSample, sampling.Rows - 1);
        Assert.Equal(sampling.SearchRows, result.Folds.FoldOf.Count);   // the folds are the sample's
        Assert.True(sampling.RefitRows == sampling.Rows || sampling.RefitRows == sampling.SearchRows);
        Assert.True(sampling.ProbeSeconds > 0);
        Assert.NotEmpty(result.Trials);
        Assert.Equal(sampling.Rows, result.Model.Transform(data).GetColumn<bool>("PredictedLabel").Count());
    }

    [Fact]
    public void ATableTooBigForItsBudget_IsSearchedWhole_WhenSamplingIsOff()
    {
        var ml = new MLContext(seed: 1);
        var (data, roles) = Table(ml, SearchSample.ProbeRows);
        var result = new TrialRunner(Request(ml, TimeSpan.FromSeconds(1), sample: false, maxTrials: 1)).Run(data, roles);
        Assert.Null(result.Sampling);
        Assert.Equal(SearchSample.ProbeRows, result.Folds.FoldOf.Count);
    }
}
