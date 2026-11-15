namespace MLoop.Extensibility.Hooks;

/// <summary>
/// Lifecycle points where hooks can execute custom logic.
/// </summary>
public enum HookType
{
    /// <summary>
    /// Executes before AutoML training begins (<c>mloop train</c>, <c>POST /train</c>).
    /// Use for: data validation, preprocessing verification, configuration checks
    /// Context: Has access to training data, configuration, metadata
    /// </summary>
    PreTrain,

    /// <summary>
    /// Executes after AutoML training completes (<c>mloop train</c>, <c>POST /train</c>).
    /// Use for: model evaluation, deployment triggers, logging/monitoring
    /// Context: Has access to trained model, metrics, experiment results
    /// </summary>
    PostTrain,

    /// <summary>
    /// Executes before <c>mloop predict</c> scores a table, on the table it is about to score.
    /// <c>mloop serve</c> runs no hooks, and neither do the tasks that predict without a table
    /// (forecasting) or over image folders.
    /// Use for: input validation, preprocessing coordination
    /// Context: Metadata["DataFile"] is the table; a refusal means nothing is scored (exit code 1)
    /// </summary>
    PrePredict,

    /// <summary>
    /// Executes after <c>mloop evaluate</c> has its metrics. <c>POST /evaluate</c> runs no hooks.
    /// Use for: metric analysis, result validation, reporting
    /// Context: Metrics (name → value) and Metadata["DataFile"]; a refusal fails the command (exit code 1)
    /// </summary>
    PostEvaluate
}
