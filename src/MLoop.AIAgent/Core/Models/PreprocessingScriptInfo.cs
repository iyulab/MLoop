namespace MLoop.AIAgent.Core.Models;

/// <summary>
/// Information about a generated preprocessing script
/// </summary>
public class PreprocessingScriptInfo
{
    /// <summary>
    /// Script sequence number (e.g., 1 for 01_handle_missing.cs)
    /// </summary>
    public required int Sequence { get; init; }

    /// <summary>
    /// Script name without extension (e.g., "handle_missing")
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// Full script filename (e.g., "01_handle_missing.cs")
    /// </summary>
    public required string FileName { get; init; }

    /// <summary>
    /// Description of what this script does
    /// </summary>
    public required string Description { get; init; }

    /// <summary>
    /// Generated C# source code
    /// </summary>
    public required string SourceCode { get; init; }
}

/// <summary>
/// Result of preprocessing script generation
/// </summary>
public class PreprocessingScriptGenerationResult
{
    /// <summary>
    /// List of generated scripts in execution order
    /// </summary>
    public required List<PreprocessingScriptInfo> Scripts { get; init; }

    /// <summary>
    /// Directory where scripts were saved
    /// </summary>
    public required string OutputDirectory { get; init; }

    /// <summary>
    /// Summary of preprocessing steps
    /// </summary>
    public required string Summary { get; init; }
}
