using System.Globalization;
using Microsoft.ML.Data;
using MLoop.Core.Models;

namespace MLoop.Core.Prediction;

/// <summary>
/// What a served row leaves unsaid to the model: a feature column the row does not carry, and a value in a numeric column that
/// is not a number. Both are read as missing — the product's contract for a missing number, not an error — so the model still
/// answers, and without this the answer reads as confidently as one for a complete row. The caller is told which columns and how
/// many rows, once per column, and the prediction is not refused.
/// </summary>
public static class ServedRowAudit
{
    /// <summary>Warnings for <paramref name="rows"/> (already matched to the schema's column names) against <paramref name="schema"/>.</summary>
    public static IReadOnlyList<string> Describe(IReadOnlyCollection<Dictionary<string, object>> rows, InputSchemaInfo schema)
    {
        ArgumentNullException.ThrowIfNull(rows);
        ArgumentNullException.ThrowIfNull(schema);
        var warnings = new List<string>();
        if (rows.Count == 0)
            return warnings;

        foreach (var column in schema.Columns.Where(c => c.Purpose.Equals("Feature", StringComparison.OrdinalIgnoreCase)))
        {
            var numeric = IsNumeric(SchemaDataTypes.FeatureLoadKind(column, DataKind.String));
            var missing = 0;
            var unreadable = 0;
            string? example = null;
            foreach (var row in rows)
            {
                if (!row.TryGetValue(column.Name, out var value) || Text(value) is not { Length: > 0 } text)
                {
                    missing++;
                    continue;
                }
                if (numeric && !double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out _))
                {
                    unreadable++;
                    example ??= text;
                }
            }

            if (missing > 0)
                warnings.Add($"Column '{column.Name}' is missing from {missing} of {rows.Count} row(s); it is read as a missing value.");
            if (unreadable > 0)
                warnings.Add($"Column '{column.Name}' holds a value that is not a number in {unreadable} of {rows.Count} row(s) (for example '{Clip(example!)}'); it is read as a missing value.");
        }
        return warnings;
    }

    private static string? Text(object? value) => value switch
    {
        null => null,
        float f => f.ToString(CultureInfo.InvariantCulture),
        double d => d.ToString(CultureInfo.InvariantCulture),
        decimal m => m.ToString(CultureInfo.InvariantCulture),
        int i => i.ToString(CultureInfo.InvariantCulture),
        long l => l.ToString(CultureInfo.InvariantCulture),
        bool b => b ? "1" : "0",
        _ => value.ToString(),
    };

    private static bool IsNumeric(DataKind kind) => kind is DataKind.Single or DataKind.Double or DataKind.Int16 or DataKind.Int32
        or DataKind.Int64 or DataKind.UInt16 or DataKind.UInt32 or DataKind.UInt64 or DataKind.SByte or DataKind.Byte;

    private static string Clip(string text) => text.Length <= 24 ? text : text[..24] + "…";
}
