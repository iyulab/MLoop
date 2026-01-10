using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;
using MLoop.Core.Preprocessing.Incremental.ScriptGeneration.Models;

namespace MLoop.Core.Preprocessing.Incremental.ScriptGeneration.Contracts;

/// <summary>
/// Generates reusable C# preprocessing scripts from approved rules.
/// Scripts can be compiled and reused for future preprocessing workflows.
/// </summary>
public interface IScriptGenerator
{
    /// <summary>
    /// Generates a complete C# script from approved preprocessing rules.
    /// </summary>
    /// <param name="rules">The approved preprocessing rules to convert to code.</param>
    /// <param name="options">Script generation options.</param>
    /// <returns>Complete C# script as a string.</returns>
    /// <remarks>
    /// Generated script includes:
    /// <list type="bullet">
    /// <item><description>Necessary using statements</description></item>
    /// <item><description>Class declaration with specified namespace and name</description></item>
    /// <item><description>Apply() method that applies all rules in sequence</description></item>
    /// <item><description>Helper methods for each rule type</description></item>
    /// <item><description>Optional validation and error handling</description></item>
    /// </list>
    /// </remarks>
    string GenerateScript(
        IReadOnlyList<PreprocessingRule> rules,
        ScriptGenerationOptions? options = null);

    /// <summary>
    /// Saves a generated script to a file.
    /// </summary>
    /// <param name="script">The script content to save.</param>
    /// <param name="outputPath">The file path to save to.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Task completing when the script is saved.</returns>
    Task SaveScriptAsync(
        string script,
        string outputPath,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates and saves a script in one operation.
    /// </summary>
    /// <param name="rules">The approved preprocessing rules.</param>
    /// <param name="outputPath">The file path to save to.</param>
    /// <param name="options">Script generation options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Task completing when the script is generated and saved.</returns>
    Task GenerateAndSaveAsync(
        IReadOnlyList<PreprocessingRule> rules,
        string outputPath,
        ScriptGenerationOptions? options = null,
        CancellationToken cancellationToken = default);
}
