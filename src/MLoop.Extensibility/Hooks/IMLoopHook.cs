namespace MLoop.Extensibility.Hooks;

/// <summary>
/// Interface for custom lifecycle hooks that execute at specific pipeline stages.
/// Hooks enable validation, logging, monitoring, and dynamic configuration.
/// </summary>
/// <remarks>
/// <para>
/// Lifecycle hooks allow expert users to inject custom logic at key pipeline points:
/// - <see cref="HookType.PreTrain"/>: Before training starts (data validation, config checks)
/// - <see cref="HookType.PostTrain"/>: After training completes (deployment triggers, logging)
/// - <see cref="HookType.PrePredict"/>: Before batch prediction (input validation)
/// - <see cref="HookType.PostEvaluate"/>: After model evaluation (metric analysis, reporting)
/// </para>
/// <para>
/// Hooks are discovered in `.mloop/scripts/hooks/{hook-type}/` and executed in alphabetical order.
/// Each hook can:
/// - Continue execution normally
/// - Abort with error message
/// - Modify pipeline configuration dynamically
/// </para>
/// <para>
/// Example: Data validation hook (pre-train)
/// <code>
/// public class DataValidationHook : IMLoopHook
/// {
///     public string Name => "Data Quality Check";
///
///     public async Task&lt;HookResult&gt; ExecuteAsync(HookContext ctx)
///     {
///         var rowCount = ctx.DataView.Preview(maxRows: 1000).RowView.Length;
///
///         if (rowCount &lt; 100)
///         {
///             ctx.Logger.Error($"Insufficient data: {rowCount} rows");
///             return HookResult.Abort("Training requires at least 100 rows");
///         }
///
///         ctx.Logger.Info($"✅ Data quality check passed: {rowCount} rows");
///         return HookResult.Continue();
///     }
/// }
/// </code>
/// </para>
/// <para>
/// Example: Automated deployment trigger (post-train)
/// <code>
/// public class AutoDeployHook : IMLoopHook
/// {
///     public string Name => "Auto Deploy on High Accuracy";
///
///     public async Task&lt;HookResult&gt; ExecuteAsync(HookContext ctx)
///     {
///         var metrics = ctx.Metrics as BinaryClassificationMetrics;
///
///         if (metrics.Accuracy > 0.95)
///         {
///             ctx.Logger.Info($"🚀 Accuracy {metrics.Accuracy:F3} > 0.95, triggering deployment");
///             await TriggerDeployment(ctx.ProjectRoot, ctx.GetMetadata&lt;string&gt;("ModelName"));
///         }
///
///         return HookResult.Continue();
///     }
/// }
/// </code>
/// </para>
/// </remarks>
public interface IMLoopHook
{
    /// <summary>
    /// Human-readable hook name for logging and debugging.
    /// Example: "Data Quality Check", "MLflow Logger", "Auto Deploy"
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Executes hook logic at the specified pipeline stage.
    /// </summary>
    /// <param name="context">
    /// Execution context providing:
    /// - Hook type and stage information
    /// - ML.NET context and data access
    /// - Training/prediction data (when available)
    /// - Trained model and metrics (post-training/evaluation)
    /// - Logger for progress and debugging
    /// - Metadata from configuration and previous hooks
    /// </param>
    /// <returns>
    /// Result indicating action to take:
    /// - <see cref="HookResult.Continue(string?)"/>: Continue normal execution (default)
    /// - <see cref="HookResult.Abort"/>: Stop execution with error message
    /// - <see cref="HookResult.ModifyConfig"/>: Modify configuration and continue
    /// </returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when hook logic fails critically. Hooks should use HookResult.Abort() for
    /// graceful failures and throw exceptions only for unexpected errors.
    /// </exception>
    Task<HookResult> ExecuteAsync(HookContext context);
}
