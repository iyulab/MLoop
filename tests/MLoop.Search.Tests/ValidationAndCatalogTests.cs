using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Search.Candidates;
using MLoop.Search.Validation;

namespace MLoop.Search.Tests;

/// <summary>
/// The validator's folds (stratified, seeded, averaged — or one holdout on large data) and the binary
/// candidate list: every trainer builds from its starting point and fits.
/// </summary>
public sealed class ValidationAndCatalogTests
{
    private sealed class Row
    {
        public bool Label { get; set; }

        [VectorType(2)]
        public float[] Features { get; set; } = [];
    }

    private static IDataView Data(MLContext ml, int n, double positiveShare = 0.3, int seed = 3)
    {
        var random = new Random(seed);
        var rows = Enumerable.Range(0, n).Select(_ =>
        {
            var positive = random.NextDouble() < positiveShare;
            var x = (float)(random.NextDouble() + (positive ? 0.8 : 0));
            return new Row { Label = positive, Features = [x, (float)random.NextDouble()] };
        }).ToList();
        return ml.Data.LoadFromEnumerable(rows);
    }

    [Fact]
    public void Folds_AreStratified_Even_AndSetByTheSeed()
    {
        var keys = Enumerable.Range(0, 103).Select(i => i % 10 < 2 ? "rare" : "common").ToArray();

        var folds = FoldPlan.Assign(keys, stratify: true, parts: 5, seed: 9);

        for (var f = 0; f < 5; f++)
        {
            var rows = Enumerable.Range(0, keys.Length).Where(i => folds[i] == f).ToList();
            Assert.InRange(rows.Count, 20, 21);
            Assert.InRange(rows.Count(i => keys[i] == "rare"), 4, 5);   // 21 rare rows over 5 folds
        }
        Assert.Equal(folds, FoldPlan.Assign(keys, stratify: true, parts: 5, seed: 9));
        Assert.NotEqual(folds, FoldPlan.Assign(keys, stratify: true, parts: 5, seed: 10));
    }

    [Fact]
    public void AHoldoutsSlices_AreStratifiedRandomShares_WhateverOrderTheFileIsIn()
    {
        // A file sorted by its label: every negative, then every positive.
        var ml = new MLContext(seed: 1);
        var rows = Enumerable.Range(0, FoldPlan.HoldoutThreshold).Select(i => new Row { Label = i >= 12_000, Features = [i, 0] }).ToList();
        var plan = FoldPlan.For(ml.Data.LoadFromEnumerable(rows), "Label", stratify: true, seed: 4);

        Assert.True(plan.IsHoldout);
        var holdout = Enumerable.Range(0, rows.Count).Where(i => plan.FoldOf[i] == 0).ToList();
        Assert.Equal(holdout.Count, plan.HoldoutSliceOf!.Count);
        for (var slice = 0; slice < Validator.HoldoutSlices; slice++)
        {
            var members = holdout.Where((_, k) => plan.HoldoutSliceOf[k] == slice).ToList();
            Assert.InRange(members.Count, 299, 301);                                 // 1,500 holdout rows over 5
            Assert.InRange(members.Count(i => rows[i].Label), 59, 61);               // a fifth of 300 positives in each
        }
    }

    [Fact]
    public void Validate_AveragesEveryFold_AndEachFoldScoresOnlyItsOwnRows()
    {
        var ml = new MLContext(seed: 1);
        var data = Data(ml, 200);
        var plan = FoldPlan.For(data, "Label", stratify: true, seed: 4);
        var candidate = CandidateCatalog.Binary().Single(c => c.Name == "LbfgsLogisticRegression");
        var pipeline = candidate.Build(ml, candidate.Space.PointAt(candidate.Space.Default), new TrainerColumns("Label", "Features"), new SearchSettings());
        var scoredRows = 0;

        var score = Validator.Validate(ml, data, plan, pipeline, scored =>
        {
            scoredRows += scored.GetColumn<bool>("Label").Count();
            var m = ml.BinaryClassification.Evaluate(scored);
            return new Dictionary<string, double> { ["accuracy"] = m.Accuracy, ["auc"] = m.AreaUnderRocCurve };
        });

        Assert.Equal(5, score.Folds);
        Assert.False(score.IsHoldout);
        Assert.Equal(200, scoredRows);                         // every row validated exactly once
        Assert.Equal(score.FoldMetrics.Average(f => f["auc"]), score.Mean["auc"], 12);
        Assert.True(score.Mean["auc"] > 0.7, $"auc {score.Mean["auc"]}");
        Assert.True(score.StandardDeviation["auc"] >= 0);
        Assert.Equal(score.StandardDeviation["auc"] / Math.Sqrt(5), score.StandardError!["auc"], 12);
    }

    /// <summary>
    /// A holdout has no folds to spread over: its standard error is the spread of five equal slices of it over √5 —
    /// so the search can tell a real lead from noise on large data too.
    /// </summary>
    [Fact]
    public void AHoldoutsStandardError_IsTheSpreadOfItsSlices()
    {
        var ml = new MLContext(seed: 1);
        var data = Data(ml, FoldPlan.HoldoutThreshold);
        var plan = FoldPlan.For(data, "Label", stratify: true, seed: 4);
        var candidate = CandidateCatalog.Binary().Single(c => c.Name == "LbfgsLogisticRegression");
        var pipeline = candidate.Build(ml, candidate.Space.PointAt(candidate.Space.Default), new TrainerColumns("Label", "Features"), new SearchSettings());
        var scoredRows = new List<int>();

        var score = Validator.Validate(ml, data, plan, pipeline, scored =>
        {
            scoredRows.Add(scored.GetColumn<bool>("Label").Count());
            return new Dictionary<string, double> { ["auc"] = ml.BinaryClassification.EvaluateNonCalibrated(scored).AreaUnderRocCurve };
        });

        Assert.True(score.IsHoldout);
        Assert.Equal(1, score.Folds);
        Assert.Equal(0, score.StandardDeviation["auc"]);
        // The holdout once, then each of its five slices.
        Assert.Equal([1500, 300, 300, 300, 300, 300], scoredRows);
        Assert.InRange(score.StandardError!["auc"], 1e-6, 0.05);
    }

    [Fact]
    public void TheSwitchThatStopsAFitInProgress_IsFound()
    {
        // ML.NET keeps it non-public; an upgrade that moves it must fail here rather than leave every
        // search stopping only between folds.
        Assert.True(FitCancellation.IsSupported);
    }

    [Fact]
    public void Validate_StopsAFitInProgress_WhenCancelled()
    {
        // A fit that cannot end on its own soon (UnendingSdcaLogisticRegression). Cancelled well before the
        // end, the fit itself must stop: a check between folds would let the first fold run on.
        var ml = new MLContext(seed: 1);
        var data = Data(ml, 60);
        var plan = FoldPlan.For(data, "Label", stratify: true, seed: 4);
        var pipeline = ml.BinaryClassification.Trainers.UnendingSdcaLogisticRegression();
        using var cut = new CancellationTokenSource(TimeSpan.FromMilliseconds(300));

        var scoredFolds = 0;

        Assert.ThrowsAny<OperationCanceledException>(() => Validator.Validate(ml, data, plan, pipeline, scored =>
        {
            scoredFolds++;
            return new Dictionary<string, double> { ["accuracy"] = ml.BinaryClassification.Evaluate(scored).Accuracy };
        }, cut.Token));
        Assert.Equal(0, scoredFolds);   // the first fold's fit was stopped, not waited out
    }

    [Fact]
    public void ALargePartition_IsScoredOnOneHoldoutOfATenth()
    {
        var ml = new MLContext(seed: 1);
        var data = Data(ml, FoldPlan.HoldoutThreshold);

        var plan = FoldPlan.For(data, "Label", stratify: true, seed: 4);

        Assert.True(plan.IsHoldout);
        Assert.Equal(1, plan.Folds);
        Assert.Equal(FoldPlan.HoldoutThreshold / 10, plan.FoldOf.Count(f => f == 0));
    }

    [Fact]
    public void EveryBinaryCandidate_BuildsFromItsStartAndFits()
    {
        var ml = new MLContext(seed: 1);
        var data = Data(ml, 300);
        var columns = new TrainerColumns("Label", "Features");

        foreach (var candidate in CandidateCatalog.Binary())
        {
            var estimator = candidate.Build(ml, candidate.Space.PointAt(candidate.Space.Default), columns, new SearchSettings());
            ITransformer model;
            try
            {
                model = estimator.Fit(data);
            }
            catch (Exception ex) when (candidate.Name == "LightGbm" && ex.GetBaseException() is DllNotFoundException)
            {
                continue;   // no native LightGBM on this platform (osx-arm64) — what isAvailable leaves out
            }
            // A start is cheap, not good: SDCA's (L1 = 1) regularizes every weight to zero — the tuner moves on from it.
            var auc = ml.BinaryClassification.EvaluateNonCalibrated(model.Transform(data)).AreaUnderRocCurve;
            Assert.True(auc >= 0.5, $"{candidate.Name}: auc {auc}");
        }
        Assert.Equal(["FastTree", "FastForest", "LightGbm", "LbfgsLogisticRegression", "SdcaLogisticRegression"],
            CandidateCatalog.Binary().Select(c => c.Name));
        Assert.DoesNotContain(CandidateCatalog.Binary(name => name != "LightGbm"), c => c.Name == "LightGbm");
    }

    private sealed class ClassRow
    {
        public string Label { get; set; } = "";

        [VectorType(2)]
        public float[] Features { get; set; } = [];
    }

    [Fact]
    public void EveryMulticlassCandidate_BuildsFromItsStart_FitsAndAnswersWithTheLabelsValues()
    {
        var ml = new MLContext(seed: 1);
        var random = new Random(5);
        string[] classes = ["low", "mid", "high"];
        var data = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 300).Select(i =>
        {
            var k = i % 3;
            return new ClassRow { Label = classes[k], Features = [(float)(k + random.NextDouble()), (float)random.NextDouble()] };
        }).ToList());
        var columns = new TrainerColumns("Label", "Features");

        foreach (var candidate in CandidateCatalog.Multiclass())
        {
            var estimator = candidate.Build(ml, candidate.Space.PointAt(candidate.Space.Default), columns, new SearchSettings());
            ITransformer model;
            try
            {
                model = estimator.Fit(data);
            }
            catch (Exception ex) when (candidate.Name == "LightGbm" && ex.GetBaseException() is DllNotFoundException)
            {
                continue;   // no native LightGBM on this platform (osx-arm64)
            }
            var scored = model.Transform(data);
            var predicted = scored.GetColumn<ReadOnlyMemory<char>>("PredictedLabel").Select(v => v.ToString()).ToHashSet();
            Assert.Subset(classes.ToHashSet(), predicted);
            // The label column comes out as the key the trainer read, so the evaluator reads it directly. A
            // start is cheap, not good: on features scaled to [0, 1], SDCA's start (L1 = 1) sets every weight to
            // zero and answers at chance — the tuner moves on from it. The floor is chance among three.
            var accuracy = ml.MulticlassClassification.Evaluate(scored, "Label").MicroAccuracy;
            Assert.True(accuracy >= 1.0 / 3 - 1e-9, $"{candidate.Name}: accuracy {accuracy}");
        }
        Assert.Equal(["FastTreeOva", "FastForestOva", "LightGbm", "LbfgsMaximumEntropy", "SdcaMaximumEntropy",
                "LbfgsLogisticRegressionOva", "SdcaLogisticRegressionOva"],
            CandidateCatalog.Multiclass().Select(c => c.Name));
    }

    private sealed class WeightedClassRow
    {
        public string Label { get; set; } = "";
        public float W { get; set; }

        [VectorType(2)]
        public float[] Features { get; set; } = [];
    }

    /// <summary>
    /// Every multiclass candidate trains on the row weights — one-versus-all included, whose binary trainers
    /// carry the weight column. Each point appears once per class, so only the weights tell the classes apart.
    /// </summary>
    [Fact]
    public void EveryMulticlassCandidate_TrainsOnTheWeights()
    {
        var ml = new MLContext(seed: 1);
        var random = new Random(7);
        string[] classes = ["a", "b", "c"];
        var rows = new List<WeightedClassRow>();
        for (var i = 0; i < 300; i++)
        {
            var x = (float)random.NextDouble();
            var region = x < 1f / 3 ? 0 : x < 2f / 3 ? 1 : 2;
            foreach (var (label, k) in classes.Select((c, k) => (c, k)))
                rows.Add(new WeightedClassRow { Label = label, W = k == region ? 20f : 1f, Features = [x, (float)random.NextDouble()] });
        }
        var data = ml.Data.LoadFromEnumerable(rows);
        var probe = ml.Data.LoadFromEnumerable(new[]
        {
            new WeightedClassRow { Label = "a", Features = [0.1f, 0.5f] },
            new WeightedClassRow { Label = "a", Features = [0.9f, 0.5f] },
        });
        var columns = new TrainerColumns("Label", "Features", "W");

        foreach (var candidate in CandidateCatalog.Multiclass(name => name != "LightGbm" || LightGbmLoads()))
        {
            if (candidate.Name.StartsWith("Sdca", StringComparison.Ordinal))
                continue;   // SDCA's start (L1 = 1) barely learns anything, weighted or not
            var model = candidate.Build(ml, candidate.Space.PointAt(candidate.Space.Default), columns, new SearchSettings()).Fit(data);
            var answers = model.Transform(probe).GetColumn<ReadOnlyMemory<char>>("PredictedLabel").Select(v => v.ToString()).ToArray();
            Assert.True(answers is ["a", "c"], $"{candidate.Name}: answered {string.Join(", ", answers)} for the low and high regions");
        }
    }

    private static bool LightGbmLoads() => System.Runtime.InteropServices.NativeLibrary.TryLoad(
        "lib_lightgbm", typeof(Microsoft.ML.Trainers.LightGbm.LightGbmMulticlassTrainer).Assembly, null, out _);

    private sealed class ValueRow
    {
        public float Label { get; set; }

        [VectorType(2)]
        public float[] Features { get; set; } = [];
    }

    [Fact]
    public void EveryRegressionCandidate_BuildsFromItsStartAndFits()
    {
        var ml = new MLContext(seed: 1);
        var random = new Random(9);
        var data = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 300).Select(_ =>
        {
            var x = (float)random.NextDouble();
            return new ValueRow { Label = 3 * x + 1 + (float)random.NextDouble() * 0.2f, Features = [x, (float)random.NextDouble()] };
        }).ToList());
        var columns = new TrainerColumns("Label", "Features");

        foreach (var candidate in CandidateCatalog.Regression(name => name != "LightGbm" || LightGbmLoads()))
        {
            var model = candidate.Build(ml, candidate.Space.PointAt(candidate.Space.Default), columns, new SearchSettings()).Fit(data);
            var metrics = ml.Regression.Evaluate(model.Transform(data));
            // A start is cheap, not good — four trees, SDCA's L1 = 1 — so only a finite score is asked of it.
            Assert.True(double.IsFinite(metrics.RootMeanSquaredError), $"{candidate.Name}: rmse {metrics.RootMeanSquaredError}");
        }
        Assert.Equal(["FastTree", "FastForest", "LightGbm", "Sdca", "LbfgsPoissonRegression"],
            CandidateCatalog.Regression().Select(c => c.Name));
    }
}
