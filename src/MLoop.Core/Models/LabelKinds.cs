using Microsoft.ML.Data;

namespace MLoop.Core.Models;

/// <summary>
/// The type a label column is loaded as, from the type the saved schema recorded for it — one rule for
/// every load of a model's data: both partitions of a training run, <c>mloop evaluate</c> and
/// <c>mloop predict</c>.
/// </summary>
/// <remarks>
/// <para>
/// The fitted pipeline maps the label to a key over the type it was trained on, so a load that types the
/// label differently cannot be scored: <c>Could not apply a map over type 'String' to column 'class'
/// since it has type 'Single'</c>. Inferring the type from each file is how that happens — inference reads
/// the first ten thousand rows, and in a file sorted by its label those rows can hold only class
/// <c>0</c> (read as a yes/no column, then a string) in one partition and <c>0</c>, <c>1</c>, <c>2</c>
/// (read as a number) in the other. The saved schema was captured once, from the whole file; it is the
/// decision every load follows.
/// </para>
/// <para>
/// A multiclass label keeps its recorded kind: class ids recorded as <c>Numeric</c> load as numbers,
/// because models were trained that way and must keep loading; categories load as strings.
/// </para>
/// </remarks>
public static class LabelKinds
{
    /// <summary>The label's <see cref="DataKind"/> for <paramref name="task"/>, given the saved schema's type name.</summary>
    public static DataKind For(string task, string? recordedDataType)
    {
        // Folded, because the task recorded in an experiment's config predates the merge-time
        // canonicalization for anything trained before it — an unrecognized spelling would fall to the
        // default arm and silently pick the wrong kind.
        return TaskTypes.Canonical(task) switch
        {
            "binary-classification" => recordedDataType == SchemaDataTypes.Boolean ? DataKind.Boolean : DataKind.String,
            "regression" or "forecasting" => DataKind.Single,
            _ => SchemaDataTypes.ToDataKind(recordedDataType ?? SchemaDataTypes.Text, fallback: DataKind.String),
        };
    }

    /// <summary>
    /// Whether training loads its partitions with the recorded kind. The classifiers that map a label to
    /// a key do; binary classification converts its own labels to yes/no after loading, and regression
    /// reads a number either way.
    /// </summary>
    public static bool AppliesAtTraining(string task) =>
        TaskTypes.Canonical(task) is "multiclass-classification" or "text-classification";
}
