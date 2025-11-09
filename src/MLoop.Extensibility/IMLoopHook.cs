namespace MLoop.Extensibility;

/// <summary>
/// Defines a lifecycle hook for custom logic at specific pipeline points.
/// Hooks allow optional code-based customization of the AutoML workflow.
/// </summary>
/// <remarks>
/// <para>
/// Hooks are executed at specific lifecycle points during ML operations:
/// </para>
/// <list type="bullet">
/// <item><description>pre-train: Before AutoML training begins</description></item>
/// <item><description>post-train: After AutoML training completes</description></item>
/// <item><description>pre-predict: Before batch prediction</description></item>
/// <item><description>post-evaluate: After model evaluation</description></item>
/// </list>
/// <para>
/// Hooks are completely optional and discovered automatically from .mloop/scripts/hooks/ directory.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// public class DataValidationHook : IMLoopHook
/// {
///     public string Name => "Data Quality Check";
///
///     public async Task&lt;HookResult&gt; ExecuteAsync(HookContext ctx)
///     {
///         var rowCount = ctx.DataView.Preview().RowView.Length;
///
///         if (rowCount &lt; 100)
///         {
///             return HookResult.Abort("Insufficient data: need at least 100 rows");
///         }
///
///         ctx.Logger.Info($"✅ Validation passed: {rowCount} rows");
///         return HookResult.Continue();
///     }
/// }
/// </code>
/// </example>
public interface IMLoopHook
{
    /// <summary>
    /// Gets the display name of this hook.
    /// Used for logging and user feedback.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Executes the hook logic at the appropriate lifecycle point.
    /// </summary>
    /// <param name="context">
    /// The execution context providing access to data, ML context, logger, and metadata.
    /// </param>
    /// <returns>
    /// A <see cref="HookResult"/> indicating whether to continue or abort the operation.
    /// - Return <see cref="HookResult.Continue"/> to proceed normally
    /// - Return <see cref="HookResult.Abort"/> to stop the operation with a reason
    /// </returns>
    /// <exception cref="Exception">
    /// Exceptions thrown by hooks are caught and logged as warnings.
    /// The operation continues with AutoML to ensure graceful degradation.
    /// </exception>
    Task<HookResult> ExecuteAsync(HookContext context);
}
