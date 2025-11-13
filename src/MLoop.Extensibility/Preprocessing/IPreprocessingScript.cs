namespace MLoop.Extensibility.Preprocessing;

/// <summary>
/// Interface for custom data preprocessing scripts.
/// Scripts are executed sequentially before AutoML training.
/// </summary>
/// <remarks>
/// <para>
/// Preprocessing scripts allow users to transform raw CSV data through custom C# logic:
/// - Multi-file operations (join, merge, concat)
/// - Data transformations (wide-to-long, feature engineering)
/// - Complex cleaning beyond FilePrepper capabilities
/// </para>
/// <para>
/// Scripts are discovered in `.mloop/scripts/preprocess/` and executed in alphabetical order:
/// 01_*.cs → 02_*.cs → 03_*.cs, with each script's output becoming the next script's input.
/// </para>
/// <para>
/// Example: Multi-file join
/// <code>
/// public class JoinMachineOrders : IPreprocessingScript
/// {
///     public async Task&lt;string&gt; ExecuteAsync(PreprocessContext context)
///     {
///         var machines = await context.Csv.ReadAsync("datasets/raw/machines.csv");
///         var orders = await context.Csv.ReadAsync("datasets/raw/orders.csv");
///
///         var joined = from m in machines
///                      join o in orders on m["item_id"] equals o["item_id"]
///                      select new Dictionary&lt;string, string&gt;
///                      {
///                          ["machine"] = m["machine"],
///                          ["item"] = m["item"],
///                          ["quantity"] = o["quantity"]
///                      };
///
///         return await context.Csv.WriteAsync(
///             Path.Combine(context.OutputDirectory, "01_joined.csv"),
///             joined.ToList());
///     }
/// }
/// </code>
/// </para>
/// </remarks>
public interface IPreprocessingScript
{
    /// <summary>
    /// Executes preprocessing logic on input CSV data.
    /// </summary>
    /// <param name="context">
    /// Execution context providing:
    /// - Input CSV path from previous script or user
    /// - Output directory for intermediate files
    /// - CSV helper for reading/writing operations
    /// - FilePrepper integration for advanced preprocessing
    /// - Logger for progress and debugging
    /// </param>
    /// <returns>
    /// Path to output CSV file that will be passed to the next script or AutoML training.
    /// Must be a valid CSV file path that exists after execution.
    /// </returns>
    /// <exception cref="PreprocessingException">
    /// Thrown when preprocessing logic fails (file not found, invalid data, etc.).
    /// </exception>
    Task<string> ExecuteAsync(PreprocessContext context);
}
