namespace MLoop.Core.Models;

/// <summary>
/// Input schema information captured during training
/// </summary>
public class InputSchemaInfo
{
    public required List<ColumnSchema> Columns { get; init; }
    public required DateTime CapturedAt { get; init; }
}

/// <summary>
/// Individual column schema information
/// </summary>
public class ColumnSchema
{
    public required string Name { get; init; }
    public required string DataType { get; init; }
    public required string Purpose { get; init; } // "Label", "Feature", "Ignore", "Exclude"

    /// <summary>
    /// For categorical columns: list of all unique values seen during training
    /// This is critical for preventing dimension mismatch during prediction
    /// </summary>
    public List<string>? CategoricalValues { get; init; }

    /// <summary>
    /// Total number of unique values (for validation)
    /// </summary>
    public int? UniqueValueCount { get; init; }
}
