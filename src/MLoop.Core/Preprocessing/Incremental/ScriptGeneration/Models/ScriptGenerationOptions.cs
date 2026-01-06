namespace MLoop.Core.Preprocessing.Incremental.ScriptGeneration.Models;

/// <summary>
/// Configuration options for generating preprocessing scripts.
/// </summary>
public sealed class ScriptGenerationOptions
{
    /// <summary>
    /// Whether to include descriptive comments in the generated script.
    /// </summary>
    public bool IncludeComments { get; init; } = true;

    /// <summary>
    /// Whether to include validation and error handling code.
    /// </summary>
    public bool IncludeValidation { get; init; } = true;

    /// <summary>
    /// Whether to include statistical logging for applied rules.
    /// </summary>
    public bool IncludeLogging { get; init; } = false;

    /// <summary>
    /// Namespace for the generated class.
    /// </summary>
    public string Namespace { get; init; } = "MLoop.Generated";

    /// <summary>
    /// Class name for the generated preprocessing script.
    /// </summary>
    public string ClassName { get; init; } = "PreprocessingScript";

    /// <summary>
    /// Target .NET version for code generation (e.g., "net10.0", "net8.0").
    /// </summary>
    public string TargetFramework { get; init; } = "net10.0";

    /// <summary>
    /// Whether to generate async methods.
    /// </summary>
    public bool GenerateAsync { get; init; } = false;

    /// <summary>
    /// Whether to make the generated class sealed.
    /// </summary>
    public bool SealedClass { get; init; } = true;
}
