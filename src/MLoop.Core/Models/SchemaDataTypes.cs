using Microsoft.ML.Data;

namespace MLoop.Core.Models;

/// <summary>
/// Single-source authority for the <see cref="ColumnSchema.DataType"/> vocabulary
/// (upstream-008). Producers (CLI TrainingEngine's schema capture, Core
/// AutoMLRunner.CaptureInputSchema) and consumers (PredictionService, CsvDataLoader,
/// CategoricalMapper) must all speak these names — before this authority existed the
/// two producers drifted (raw .NET type names vs semantic names), so a schema captured
/// in-process resolved every numeric feature to <see cref="DataKind.String"/> and
/// model.Transform threw. Same drift class as ExperimentLayout (F-33) and
/// MetricDirection (F-27); see CLAUDE.md "Single-Source Authorities".
/// </summary>
public static class SchemaDataTypes
{
    public const string Numeric = "Numeric";
    public const string Categorical = "Categorical";
    public const string Text = "Text";
    public const string Boolean = "Boolean";

    /// <summary>
    /// Exclusion markers: the <see cref="ColumnSchema.DataType"/> a column carries when
    /// featurization dropped it (<see cref="ColumnSchema.Purpose"/> is then "Exclude").
    /// The reason is part of the same vocabulary because the saved schema is what predict and
    /// evaluate replay — see <see cref="Data.CsvDataLoader.DetermineExcludedColumns"/>.
    /// </summary>
    public const string ExcludedDateTime = "DateTime";

    /// <inheritdoc cref="ExcludedDateTime"/>
    public const string ExcludedSparse = "Sparse";

    /// <inheritdoc cref="ExcludedDateTime"/>
    public const string ExcludedConstant = "Constant";

    /// <summary>
    /// Producer-side mapping: raw CLR type of a DataView column (or vector item) to the
    /// semantic vocabulary. String-typed columns map to <see cref="Text"/> — the producer
    /// upgrades to <see cref="Categorical"/> when it captures the column's distinct values.
    /// </summary>
    public static string FromRawType(Type rawType)
    {
        if (rawType == typeof(float) || rawType == typeof(double) ||
            rawType == typeof(int) || rawType == typeof(long) ||
            rawType == typeof(short) || rawType == typeof(byte) || rawType == typeof(sbyte) ||
            rawType == typeof(uint) || rawType == typeof(ulong) || rawType == typeof(ushort))
            return Numeric;

        if (rawType == typeof(bool))
            return Boolean;

        return Text;
    }

    /// <summary>
    /// Consumer-side mapping: a persisted <see cref="ColumnSchema.DataType"/> name to the
    /// <see cref="DataKind"/> a TextLoader must load that column as. Tolerates the raw .NET
    /// type names that schemas captured before the vocabulary was unified persisted
    /// ("String" tolerance predates this — BUG-42). Unknown names return
    /// <paramref name="fallback"/> so callers keep their inferred kind.
    /// </summary>
    public static DataKind ToDataKind(string dataType, DataKind fallback) => dataType switch
    {
        Numeric => DataKind.Single,
        Categorical => DataKind.String,
        Text => DataKind.String,
        Boolean => DataKind.Boolean,
        // Legacy raw .NET names (pre-unification schemas):
        "Single" => DataKind.Single,
        "Double" => DataKind.Double,
        "Int32" => DataKind.Int32,
        "Int64" => DataKind.Int64,
        "String" => DataKind.String,
        _ => fallback
    };
}
