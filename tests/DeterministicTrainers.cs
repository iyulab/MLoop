using Microsoft.ML;
using Microsoft.ML.Trainers;

/// <summary>
/// The SDCA trainers a test fixture builds a model with, pinned to one thread and a bounded number
/// of passes.
/// </summary>
/// <remarks>
/// <para>
/// ML.NET's SDCA trainers default to several threads on a machine with more than a couple of cores.
/// On the tiny fixtures these suites train (twenty rows is typical), the multi-threaded solver does
/// not converge: it runs until its iteration limit, and its result differs from run to run even
/// with a seeded context. On a healthy machine that cost a few seconds and hid; on a starved one it
/// ran for many minutes per fit, and three suites stopped at exactly these fits — measured on the
/// same fixture, one thread finished 300 fits with the slowest at 9.5 s while the default did not
/// finish one in 60 s. One thread gives the same model every time; the iteration cap (below) bounds
/// the fixtures that cannot converge at all.
/// </para>
/// <para>
/// Linked into every test assembly from one file, like the output culture pin. A contract test
/// refuses a bare SDCA call anywhere under <c>tests/</c>, so a new fixture cannot bring the default
/// back.
/// </para>
/// </remarks>
internal static class DeterministicTrainers
{
    /// <summary>
    /// Enough passes for a twenty-row fixture to settle, and a bound on the ones that never do: a
    /// linearly separable fixture has no logistic-regression optimum (the weights grow without
    /// limit), so without a cap even one thread runs to the solver's own heuristic limit — measured,
    /// over 90 s for one fit on a starved machine, against 12 s with this cap.
    /// </summary>
    private const int MaximumIterations = 100;

    public static SdcaRegressionTrainer SingleThreadSdca(
        this RegressionCatalog.RegressionTrainers trainers,
        string labelColumnName = "Label",
        string featureColumnName = "Features")
        => trainers.Sdca(new SdcaRegressionTrainer.Options
        {
            LabelColumnName = labelColumnName,
            FeatureColumnName = featureColumnName,
            NumberOfThreads = 1,
            MaximumNumberOfIterations = MaximumIterations
        });

    public static SdcaLogisticRegressionBinaryTrainer SingleThreadSdcaLogisticRegression(
        this BinaryClassificationCatalog.BinaryClassificationTrainers trainers,
        string labelColumnName = "Label",
        string featureColumnName = "Features")
        => trainers.SdcaLogisticRegression(new SdcaLogisticRegressionBinaryTrainer.Options
        {
            LabelColumnName = labelColumnName,
            FeatureColumnName = featureColumnName,
            NumberOfThreads = 1,
            MaximumNumberOfIterations = MaximumIterations
        });

    /// <summary>
    /// The opposite of the helpers above, for a test about stopping a fit: one thread, a million passes and
    /// no convergence exit, so the fit runs for seconds whatever the rows — the shape of SDCA's own
    /// 30,000-pass default on rows it does not converge on, without depending on which rows those are.
    /// </summary>
    public static SdcaLogisticRegressionBinaryTrainer UnendingSdcaLogisticRegression(
        this BinaryClassificationCatalog.BinaryClassificationTrainers trainers,
        string labelColumnName = "Label",
        string featureColumnName = "Features")
        => trainers.SdcaLogisticRegression(new SdcaLogisticRegressionBinaryTrainer.Options
        {
            LabelColumnName = labelColumnName,
            FeatureColumnName = featureColumnName,
            NumberOfThreads = 1,
            MaximumNumberOfIterations = 1_000_000,
            ConvergenceTolerance = 1e-30f,
        });

    public static SdcaMaximumEntropyMulticlassTrainer SingleThreadSdcaMaximumEntropy(
        this MulticlassClassificationCatalog.MulticlassClassificationTrainers trainers,
        string labelColumnName = "Label",
        string featureColumnName = "Features")
        => trainers.SdcaMaximumEntropy(new SdcaMaximumEntropyMulticlassTrainer.Options
        {
            LabelColumnName = labelColumnName,
            FeatureColumnName = featureColumnName,
            NumberOfThreads = 1,
            MaximumNumberOfIterations = MaximumIterations
        });
}
