using MLoop.Extensibility.Preprocessing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

/// <summary>
/// Common data cleaning operations:
/// - Remove duplicate rows
/// - Trim whitespace from every cell
/// - (Optional) impute missing values — see the note inside for where to plug your schema in
///
/// Usage: Place in .mloop/scripts/preprocess/04_data_cleaning.cs
/// Executes automatically before training when present.
///
/// Contract: IPreprocessingScript.ExecuteAsync returns the path to the produced CSV.
/// Use the injected <see cref="ICsvHelper"/> (ctx.Csv) for encoding-aware read/write — no external
/// CSV library or temp-path helpers are needed.
/// </summary>
public class DataCleaningScript : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext ctx)
    {
        ctx.Logger.Info("🧹 Data Cleaning: Removing duplicates, trimming whitespace");

        // Read input CSV as a list of column→value rows.
        var rows = await ctx.Csv.ReadAsync(ctx.InputPath);
        ctx.Logger.Info($"Loaded: {rows.Count:N0} rows");

        // 1. Trim whitespace from every cell (prevents "Yes " != "Yes" mismatches).
        foreach (var row in rows)
        {
            foreach (var key in row.Keys.ToList())
            {
                row[key] = row[key]?.Trim() ?? "";
            }
        }
        ctx.Logger.Info("  ✓ Trimmed whitespace from all columns");

        // 2. Remove duplicate rows (compare by the full ordered set of cell values).
        var seen = new HashSet<string>();
        var deduped = new List<Dictionary<string, string>>(rows.Count);
        foreach (var row in rows)
        {
            var signature = string.Join("", row.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key}={kv.Value}"));
            if (seen.Add(signature))
            {
                deduped.Add(row);
            }
        }
        ctx.Logger.Info($"  ✓ Removed {rows.Count - deduped.Count:N0} duplicate rows");

        // 3. Missing value imputation (optional):
        // Implement based on your data schema. For numeric columns, median imputation is a safe
        // default; for categorical columns, use the mode. See 06_missing_value_imputation.cs for a
        // complete example.

        // Save cleaned data into the intermediate output directory.
        var outputPath = Path.Combine(ctx.OutputDirectory, "04_cleaned.csv");
        await ctx.Csv.WriteAsync(outputPath, deduped);

        ctx.Logger.Info($"✅ Saved: {outputPath}");
        return outputPath;
    }
}
