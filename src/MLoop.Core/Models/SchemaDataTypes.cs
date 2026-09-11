using Microsoft.ML.Data;

namespace MLoop.Core.Models;

/// <summary>
/// Single-source authority for the <see cref="ColumnSchema.DataType"/> vocabulary
/// Producers (CLI TrainingEngine's schema capture, Core
/// AutoMLRunner.CaptureInputSchema) and consumers (PredictionService, CsvDataLoader,
/// CategoricalMapper) must all speak these names — before this authority existed the
/// two producers drifted (raw .NET type names vs semantic names), so a schema captured
/// in-process resolved every numeric feature to <see cref="DataKind.String"/> and
/// model.Transform threw. Same drift class as ExperimentLayout and
/// MetricDirection; see CLAUDE.md "Single-Source Authorities".
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
    /// A text column in which every row carries its own distinct, whitespace-free value — a customer
    /// id, an order number, a UUID. See <see cref="Data.CsvDataLoader.RemoveIdentifierColumns"/>.
    /// </summary>
    public const string ExcludedIdentifier = "Identifier";

    /// <summary>
    /// Every name this vocabulary defines, including the exclusion markers.
    /// </summary>
    public static IReadOnlyList<string> All { get; } =
        [Numeric, Categorical, Text, Boolean, ExcludedDateTime, ExcludedSparse, ExcludedConstant, ExcludedIdentifier];

    /// <summary>
    /// Whether <paramref name="dataType"/> is one of this vocabulary's names.
    /// </summary>
    /// <remarks>
    /// A schema carrying a name from outside it is not rejected anywhere — consumers simply fail to
    /// match on it, and the failure surfaces further downstream as a missing feature vector or a
    /// column loaded as the wrong kind. Asking this question is what lets such a consumer name the
    /// actual cause instead of reporting that symptom. Comparison is case-insensitive, the same way
    /// every consumer compares these names.
    /// </remarks>
    public static bool IsKnown(string? dataType) =>
        dataType is not null && All.Any(name => name.Equals(dataType, StringComparison.OrdinalIgnoreCase));

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
    /// Whether <paramref name="value"/> reads as <paramref name="dataType"/> — the same question
    /// the loader answers when it coerces a column to the kind training fitted on, asked here
    /// before the coercion so a caller can say which column disagreed.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Only <see cref="Numeric"/> and <see cref="Boolean"/> can disagree with a value: every other
    /// name in this vocabulary loads as text, which any field satisfies. An empty field is not a
    /// disagreement — a missing value is a missing value, and the loader has its own handling for
    /// it.
    /// </para>
    /// <para>
    /// Culture-invariant, deliberately. The value is being read the way <c>TextLoader</c> will read
    /// it, and that parse does not follow the operator's locale; judging it by a locale that
    /// accepts a comma decimal separator would report agreement where the loader will find none.
    /// </para>
    /// </remarks>
    public static bool ValueReadsAs(string dataType, string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return true;

        return ToDataKind(dataType, DataKind.String) switch
        {
            DataKind.Single or DataKind.Double =>
                double.TryParse(value, System.Globalization.NumberStyles.Float,
                                System.Globalization.CultureInfo.InvariantCulture, out _),
            DataKind.Boolean => bool.TryParse(value, out _)
                                || value is "0" or "1",
            _ => true,
        };
    }

    /// <summary>
    /// Consumer-side mapping: a persisted <see cref="ColumnSchema.DataType"/> name to the
    /// <see cref="DataKind"/> a TextLoader must load that column as. Tolerates the raw .NET
    /// type names that schemas captured before the vocabulary was unified persisted
    /// ("String" tolerance predates this). Unknown names return
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
