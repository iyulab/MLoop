using System.Text.Json.Serialization;

namespace MLoop.Core.Models;

/// <summary>
/// Input schema information captured during training
/// </summary>
public class InputSchemaInfo
{
    public required List<ColumnSchema> Columns { get; init; }
    public required DateTime CapturedAt { get; init; }

    /// <summary>
    /// This schema with <paramref name="columns"/> recorded as <c>Ignore</c>: columns the trainer read,
    /// like the label, that a prediction does not supply — a question-answering answer's start position.
    /// Left as <c>Feature</c>, prediction refused every file without them.
    /// </summary>
    public InputSchemaInfo WithTrainingOnly(IReadOnlyCollection<string> columns) => new()
    {
        CapturedAt = CapturedAt,
        Columns = Columns
            .Select(c => columns.Contains(c.Name, StringComparer.OrdinalIgnoreCase) && c.Purpose == "Feature"
                ? new ColumnSchema
                {
                    Name = c.Name,
                    DataType = c.DataType,
                    Purpose = "Ignore",
                    CategoricalValues = c.CategoricalValues,
                    UniqueValueCount = c.UniqueValueCount,
                    MajorityClassRatio = c.MajorityClassRatio,
                    ReadByName = c.ReadByName,
                }
                : c)
            .ToList()
    };

    /// <summary>
    /// This schema with <paramref name="columns"/> marked <see cref="ColumnSchema.ReadByName"/>: the columns a
    /// featurizer in the model's own pipeline names (a prep step's normalize or fill-missing), which training
    /// loaded as columns of their own rather than as slots of the merged feature vector.
    /// </summary>
    public InputSchemaInfo WithReadByName(IReadOnlyCollection<string> columns) => new()
    {
        CapturedAt = CapturedAt,
        Columns = Columns
            .Select(c => columns.Contains(c.Name, StringComparer.OrdinalIgnoreCase) && c.Purpose == "Feature"
                ? new ColumnSchema
                {
                    Name = c.Name,
                    DataType = c.DataType,
                    Purpose = c.Purpose,
                    CategoricalValues = c.CategoricalValues,
                    UniqueValueCount = c.UniqueValueCount,
                    MajorityClassRatio = c.MajorityClassRatio,
                    ReadByName = true,
                }
                : c)
            .ToList()
    };

    /// <summary>The names of the columns the model reads by their own name (<see cref="ColumnSchema.ReadByName"/>).</summary>
    [JsonIgnore]
    public IEnumerable<string> ReadByNameColumns => Columns.Where(c => c.ReadByName).Select(c => c.Name);
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

    /// <summary>
    /// The model's pipeline reads this column by its own name — a prep step's featurizer (normalize, fill-missing
    /// mean) names it — so training loaded it as a column of its own, not as a slot of the merged feature vector,
    /// and every reader after training must load it that way too. A property of training, not of the file
    /// being read. False for a schema written before this field existed, which is how those models were read.
    /// </summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool ReadByName { get; init; }
}
