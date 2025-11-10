namespace MLoop.Extensibility;

/// <summary>
/// Defines a preprocessing script for data transformation before AutoML training.
/// Preprocessing scripts enable complex data operations beyond basic CSV cleaning.
/// </summary>
/// <remarks>
/// <para>
/// Preprocessing scripts are executed sequentially in numeric order (01_*.cs → 02_*.cs → 03_*.cs)
/// before training begins. Each script receives the output of the previous script as input.
/// </para>
/// <para>
/// Common use cases:
/// </para>
/// <list type="bullet">
/// <item><description>Multi-file operations (join, merge, concat)</description></item>
/// <item><description>Wide-to-Long transformations (unpivot)</description></item>
/// <item><description>Feature engineering (computed columns)</description></item>
/// <item><description>Complex data cleaning beyond FilePrepper</description></item>
/// </list>
/// <para>
/// Scripts are discovered automatically from .mloop/scripts/preprocess/ directory.
/// They are completely optional - if no scripts exist, preprocessing is skipped.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // .mloop/scripts/preprocess/01_join_files.cs
/// public class JoinMachineAndOrder : IPreprocessingScript
/// {
///     public async Task&lt;string&gt; ExecuteAsync(PreprocessContext ctx)
///     {
///         var machines = await ctx.Csv.ReadAsync("datasets/raw/machine_info.csv");
///         var orders = await ctx.Csv.ReadAsync("datasets/raw/order_info.csv");
///
///         var joined = from m in machines
///                      join o in orders on m["item"] equals o["중산도면"]
///                      select new Dictionary&lt;string, string&gt;
///                      {
///                          ["설비명"] = m["설비명"],
///                          ["item"] = m["item"],
///                          ["재고"] = o["재고"],
///                          ["생산필요량"] = o["생산필요량"]
///                      };
///
///         return await ctx.Csv.WriteAsync(
///             Path.Combine(ctx.OutputDirectory, "01_joined.csv"),
///             joined.ToList());
///     }
/// }
/// </code>
/// </example>
public interface IPreprocessingScript
{
    /// <summary>
    /// Executes preprocessing logic on the input data.
    /// </summary>
    /// <param name="context">
    /// The execution context providing access to input/output paths, CSV helper,
    /// FilePrepper integration, and logger.
    /// </param>
    /// <returns>
    /// The absolute path to the output CSV file. This becomes the input for the next script
    /// in the sequence, or the final training data if this is the last script.
    /// </returns>
    /// <exception cref="Exception">
    /// Exceptions thrown by preprocessing scripts are caught and logged as errors.
    /// The operation is aborted to prevent training on invalid data.
    /// </exception>
    Task<string> ExecuteAsync(PreprocessContext context);
}
