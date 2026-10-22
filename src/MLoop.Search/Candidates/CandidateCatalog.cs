using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using MLoop.Search.Tuning;

namespace MLoop.Search.Candidates;

/// <summary>
/// The trainers a search tries for each task, with the search space and starting cost of each — the
/// list ML.NET AutoML searched (dotnet/machinelearning <c>cf9093abb6</c>: <c>CodeGen/*_search_space.json</c>,
/// <c>SweepableEstimator/Estimators/*.cs</c>, <c>Tuner/PipelineProposer.cs</c>), declared here.
/// </summary>
/// <remarks>
/// The spaces start cheap on purpose — 4 trees, 4 leaves — because the cost-frugal tuner grows a model
/// from a cheap start; moving the start changes the search, not just its first trial. Where AutoML left
/// the thread count to the trainer, a candidate takes <see cref="SearchSettings.Threads"/>, and LightGBM
/// runs deterministic with the seed: a search with one seed gives one result.
/// </remarks>
public static class CandidateCatalog
{
    /// <summary>
    /// The binary classification trainers: FastTree, FastForest, LightGbm, LbfgsLogisticRegression and
    /// SdcaLogisticRegression. <paramref name="isAvailable"/> leaves out a trainer whose native library
    /// the platform does not have (LightGBM has none for osx-arm64).
    /// </summary>
    public static IReadOnlyList<TrainerCandidate> Binary(Func<string, bool>? isAvailable = null)
    {
        TrainerCandidate[] all =
        [
            new("FastTree", FastTreeSpace, 3.701, (ml, p, c, s) => ml.BinaryClassification.Trainers.FastTree(new FastTreeBinaryTrainer.Options
            {
                NumberOfLeaves = p.Int("NumberOfLeaves"),
                NumberOfTrees = p.Int("NumberOfTrees"),
                MinimumExampleCountPerLeaf = p.Int("MinimumExampleCountPerLeaf"),
                LearningRate = p["LearningRate"],
                MaximumBinCountPerFeature = p.Int("MaximumBinCountPerFeature"),
                FeatureFraction = p["FeatureFraction"],
                DiskTranspose = false,
                NumberOfThreads = s.Threads,
                Seed = s.Seed,
                LabelColumnName = c.Label,
                FeatureColumnName = c.Features,
                ExampleWeightColumnName = c.Weight,
            })),
            new("FastForest", FastForestSpace, 4.283, (ml, p, c, s) => ml.BinaryClassification.Trainers.FastForest(new FastForestBinaryTrainer.Options
            {
                NumberOfTrees = p.Int("NumberOfTrees"),
                NumberOfLeaves = p.Int("NumberOfLeaves"),
                FeatureFraction = p.Single("FeatureFraction"),
                NumberOfThreads = s.Threads,
                Seed = s.Seed,
                LabelColumnName = c.Label,
                FeatureColumnName = c.Features,
                ExampleWeightColumnName = c.Weight,
            })),
            new("LightGbm", LightGbmSpace, 4.765, (ml, p, c, s) => ml.BinaryClassification.Trainers.LightGbm(new LightGbmBinaryTrainer.Options
            {
                NumberOfLeaves = p.Int("NumberOfLeaves"),
                NumberOfIterations = p.Int("NumberOfTrees"),
                MinimumExampleCountPerLeaf = p.Int("MinimumExampleCountPerLeaf"),
                LearningRate = p["LearningRate"],
                MaximumBinCountPerFeature = p.Int("MaximumBinCountPerFeature"),
                Booster = new GradientBooster.Options
                {
                    SubsampleFraction = p["SubsampleFraction"],
                    FeatureFraction = p["FeatureFraction"],
                    L1Regularization = p["L1Regularization"],
                    L2Regularization = p["L2Regularization"],
                },
                NumberOfThreads = s.Threads,
                Seed = s.Seed,
                Deterministic = true,
                LabelColumnName = c.Label,
                FeatureColumnName = c.Features,
                ExampleWeightColumnName = c.Weight,
            })),
            new("LbfgsLogisticRegression", LbfgsSpace, 11.513, (ml, p, c, s) => ml.BinaryClassification.Trainers.LbfgsLogisticRegression(new LbfgsLogisticRegressionBinaryTrainer.Options
            {
                L1Regularization = p.Single("L1Regularization"),
                L2Regularization = p.Single("L2Regularization"),
                NumberOfThreads = s.Threads,
                LabelColumnName = c.Label,
                FeatureColumnName = c.Features,
                ExampleWeightColumnName = c.Weight,
            })),
            new("SdcaLogisticRegression", SdcaSpace, 13.16, (ml, p, c, s) => ml.BinaryClassification.Trainers.SdcaLogisticRegression(new SdcaLogisticRegressionBinaryTrainer.Options
            {
                L1Regularization = p.Single("L1Regularization"),
                L2Regularization = p.Single("L2Regularization"),
                NumberOfThreads = s.Threads,
                LabelColumnName = c.Label,
                FeatureColumnName = c.Features,
                ExampleWeightColumnName = c.Weight,
            })),
        ];
        return [.. all.Where(t => isAvailable?.Invoke(t.Name) ?? true)];
    }

    /// <summary>
    /// The multiclass classification trainers: LightGbm, LbfgsMaximumEntropy and SdcaMaximumEntropy, and
    /// one-versus-all over FastTree, FastForest, LbfgsLogisticRegression and SdcaLogisticRegression — the
    /// last always searched by AutoML though its option to leave it out was never passed, so it is named
    /// here rather than inherited unseen. Each maps the label to a key ahead of the trainer and the
    /// predicted key back to the label's value after it, as AutoML's pipelines did.
    /// </summary>
    public static IReadOnlyList<TrainerCandidate> Multiclass(Func<string, bool>? isAvailable = null)
    {
        TrainerCandidate[] all =
        [
            new("FastTreeOva", FastTreeSpace, 3.701, (ml, p, c, s) => KeyedLabel(ml, c, ml.MulticlassClassification.Trainers.OneVersusAll(
                ml.BinaryClassification.Trainers.FastTree(new FastTreeBinaryTrainer.Options
                {
                    NumberOfLeaves = p.Int("NumberOfLeaves"),
                    NumberOfTrees = p.Int("NumberOfTrees"),
                    MinimumExampleCountPerLeaf = p.Int("MinimumExampleCountPerLeaf"),
                    LearningRate = p["LearningRate"],
                    MaximumBinCountPerFeature = p.Int("MaximumBinCountPerFeature"),
                    FeatureFraction = p["FeatureFraction"],
                    DiskTranspose = false,
                    NumberOfThreads = s.Threads,
                    Seed = s.Seed,
                    LabelColumnName = c.Label,
                    FeatureColumnName = c.Features,
                    ExampleWeightColumnName = c.Weight,
                }), c.Label))),
            new("FastForestOva", FastForestSpace, 4.283, (ml, p, c, s) => KeyedLabel(ml, c, ml.MulticlassClassification.Trainers.OneVersusAll(
                ml.BinaryClassification.Trainers.FastForest(new FastForestBinaryTrainer.Options
                {
                    NumberOfTrees = p.Int("NumberOfTrees"),
                    NumberOfLeaves = p.Int("NumberOfLeaves"),
                    FeatureFraction = p.Single("FeatureFraction"),
                    NumberOfThreads = s.Threads,
                    Seed = s.Seed,
                    LabelColumnName = c.Label,
                    FeatureColumnName = c.Features,
                    ExampleWeightColumnName = c.Weight,
                }), c.Label))),
            new("LightGbm", LightGbmSpace, 4.765, (ml, p, c, s) => KeyedLabel(ml, c, ml.MulticlassClassification.Trainers.LightGbm(new LightGbmMulticlassTrainer.Options
            {
                NumberOfLeaves = p.Int("NumberOfLeaves"),
                NumberOfIterations = p.Int("NumberOfTrees"),
                MinimumExampleCountPerLeaf = p.Int("MinimumExampleCountPerLeaf"),
                LearningRate = p["LearningRate"],
                MaximumBinCountPerFeature = p.Int("MaximumBinCountPerFeature"),
                Booster = new GradientBooster.Options
                {
                    SubsampleFraction = p["SubsampleFraction"],
                    FeatureFraction = p["FeatureFraction"],
                    L1Regularization = p["L1Regularization"],
                    L2Regularization = p["L2Regularization"],
                },
                NumberOfThreads = s.Threads,
                Seed = s.Seed,
                Deterministic = true,
                LabelColumnName = c.Label,
                FeatureColumnName = c.Features,
                ExampleWeightColumnName = c.Weight,
            }))),
            new("LbfgsMaximumEntropy", LbfgsSpace, 7.980, (ml, p, c, s) => KeyedLabel(ml, c, ml.MulticlassClassification.Trainers.LbfgsMaximumEntropy(new LbfgsMaximumEntropyMulticlassTrainer.Options
            {
                L1Regularization = p.Single("L1Regularization"),
                L2Regularization = p.Single("L2Regularization"),
                NumberOfThreads = s.Threads,
                LabelColumnName = c.Label,
                FeatureColumnName = c.Features,
                ExampleWeightColumnName = c.Weight,
            }))),
            new("SdcaMaximumEntropy", SdcaSpace, 10.129, (ml, p, c, s) => KeyedLabel(ml, c, ml.MulticlassClassification.Trainers.SdcaMaximumEntropy(new SdcaMaximumEntropyMulticlassTrainer.Options
            {
                L1Regularization = p.Single("L1Regularization"),
                L2Regularization = p.Single("L2Regularization"),
                NumberOfThreads = s.Threads,
                LabelColumnName = c.Label,
                FeatureColumnName = c.Features,
                ExampleWeightColumnName = c.Weight,
            }))),
            new("LbfgsLogisticRegressionOva", LbfgsSpace, 11.513, (ml, p, c, s) => KeyedLabel(ml, c, ml.MulticlassClassification.Trainers.OneVersusAll(
                ml.BinaryClassification.Trainers.LbfgsLogisticRegression(new LbfgsLogisticRegressionBinaryTrainer.Options
                {
                    L1Regularization = p.Single("L1Regularization"),
                    L2Regularization = p.Single("L2Regularization"),
                    NumberOfThreads = s.Threads,
                    LabelColumnName = c.Label,
                    FeatureColumnName = c.Features,
                    ExampleWeightColumnName = c.Weight,
                }), c.Label))),
            new("SdcaLogisticRegressionOva", SdcaSpace, 13.16, (ml, p, c, s) => KeyedLabel(ml, c, ml.MulticlassClassification.Trainers.OneVersusAll(
                ml.BinaryClassification.Trainers.SdcaLogisticRegression(new SdcaLogisticRegressionBinaryTrainer.Options
                {
                    L1Regularization = p.Single("L1Regularization"),
                    L2Regularization = p.Single("L2Regularization"),
                    NumberOfThreads = s.Threads,
                    LabelColumnName = c.Label,
                    FeatureColumnName = c.Features,
                    ExampleWeightColumnName = c.Weight,
                }), c.Label))),
        ];
        return [.. all.Where(t => isAvailable?.Invoke(t.Name) ?? true)];
    }

    // The label as a key for the trainer, and the predicted key back as the label's value.
    private static IEstimator<ITransformer> KeyedLabel<TTransformer>(MLContext ml, TrainerColumns columns, IEstimator<TTransformer> trainer)
        where TTransformer : class, ITransformer
        => ml.Transforms.Conversion.MapValueToKey(columns.Label)
            .Append(trainer)
            .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

    /// <summary>LightGBM (<c>lgbm_search_space.json</c>).</summary>
    public static SearchSpace LightGbmSpace { get; } = new([
        SearchDimension.Integer("NumberOfLeaves", 4, 32768, logBase: false, defaultValue: 4),
        SearchDimension.Integer("MinimumExampleCountPerLeaf", 20, 1024, logBase: true, defaultValue: 20),
        SearchDimension.Double("LearningRate", 2e-10, 1, logBase: true, defaultValue: 1),
        SearchDimension.Integer("NumberOfTrees", 4, 32768, logBase: false, defaultValue: 4),
        SearchDimension.Double("SubsampleFraction", 2e-10, 1, logBase: true, defaultValue: 1),
        SearchDimension.Integer("MaximumBinCountPerFeature", 8, 1024, logBase: true, defaultValue: 256),
        SearchDimension.Double("FeatureFraction", 2e-10, 1, logBase: false, defaultValue: 1),
        SearchDimension.Double("L1Regularization", 2e-10, 1, logBase: true, defaultValue: 2e-10),
        SearchDimension.Double("L2Regularization", 2e-10, 1, logBase: true, defaultValue: 1),
    ]);

    /// <summary>FastTree (<c>fast_tree_search_space.json</c>).</summary>
    public static SearchSpace FastTreeSpace { get; } = new([
        SearchDimension.Integer("NumberOfLeaves", 4, 32768, logBase: true, defaultValue: 4),
        SearchDimension.Integer("MinimumExampleCountPerLeaf", 2, 128, logBase: true, defaultValue: 20),
        SearchDimension.Integer("NumberOfTrees", 4, 32768, logBase: true, defaultValue: 4),
        SearchDimension.Integer("MaximumBinCountPerFeature", 8, 1024, logBase: true, defaultValue: 256),
        SearchDimension.Double("FeatureFraction", 2e-10, 1, logBase: false, defaultValue: 1),
        SearchDimension.Double("LearningRate", 2e-10, 1, logBase: true, defaultValue: 0.1),
    ]);

    /// <summary>FastForest (<c>fast_forest_search_space.json</c>).</summary>
    public static SearchSpace FastForestSpace { get; } = new([
        SearchDimension.Integer("NumberOfTrees", 4, 32768, logBase: true, defaultValue: 4),
        SearchDimension.Integer("NumberOfLeaves", 4, 32768, logBase: true, defaultValue: 4),
        SearchDimension.Single("FeatureFraction", 2e-10f, 1, logBase: false, defaultValue: 1),
    ]);

    /// <summary>L-BFGS (<c>lbfgs_search_space.json</c>).</summary>
    public static SearchSpace LbfgsSpace { get; } = new([
        SearchDimension.Single("L1Regularization", 0.03125f, 32768, logBase: true, defaultValue: 1),
        SearchDimension.Single("L2Regularization", 0.03125f, 32768, logBase: true, defaultValue: 1),
    ]);

    /// <summary>SDCA (<c>sdca_search_space.json</c>).</summary>
    public static SearchSpace SdcaSpace { get; } = new([
        SearchDimension.Single("L1Regularization", 0.03125f, 32768, logBase: true, defaultValue: 1),
        SearchDimension.Single("L2Regularization", 0.03125f, 32768, logBase: true, defaultValue: 0.1f),
    ]);
}
