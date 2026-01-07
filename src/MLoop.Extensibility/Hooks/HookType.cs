namespace MLoop.Extensibility.Hooks;

/// <summary>
/// Lifecycle points where hooks can execute custom logic.
/// </summary>
public enum HookType
{
    /// <summary>
    /// Executes before AutoML training begins.
    /// Use for: data validation, preprocessing verification, configuration checks
    /// Context: Has access to training data, configuration, metadata
    /// </summary>
    PreTrain,

    /// <summary>
    /// Executes after AutoML training completes.
    /// Use for: model evaluation, deployment triggers, logging/monitoring
    /// Context: Has access to trained model, metrics, experiment results
    /// </summary>
    PostTrain,

    /// <summary>
    /// Executes before batch prediction runs.
    /// Use for: input validation, preprocessing coordination
    /// Context: Has access to prediction input data
    /// </summary>
    PrePredict,

    /// <summary>
    /// Executes after model evaluation.
    /// Use for: metric analysis, result validation, reporting
    /// Context: Has access to evaluation metrics and results
    /// </summary>
    PostEvaluate
}
