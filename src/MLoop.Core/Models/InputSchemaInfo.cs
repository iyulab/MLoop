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

    /// <summary>
    /// For a classification <b>label</b>: the share of rows belonging to its most common class —
    /// the accuracy a model reaches by always predicting that class and learning nothing (the
    /// no-information rate).
    /// <para>
    /// Recorded so the promotion quality gate has a baseline that means something on imbalanced
    /// data: <c>1/UniqueValueCount</c> only describes random guessing, which is a far weaker
    /// opponent than the majority class whenever the classes are not balanced.
    /// </para>
    /// <para>
    /// Null when it was not measured — a numeric label, a regression task, a schema written before
    /// this field existed. Consumers must treat null as "unknown", never as zero.
    /// </para>
    /// </summary>
    public double? MajorityClassRatio { get; init; }
}
