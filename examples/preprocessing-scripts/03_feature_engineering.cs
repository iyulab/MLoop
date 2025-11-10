using MLoop.Extensibility;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

/// <summary>
/// Example: Feature Engineering
/// Demonstrates: Computing derived features from existing columns
/// Use case: Creating production efficiency and inventory metrics
/// Sequential: This script expects input from previous preprocessing steps
/// </summary>
public class FeatureEngineering : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext ctx)
    {
        ctx.Logger.Info("=== Feature Engineering ===");

        // This is the 3rd script (03_*.cs), so it receives output from 02_*.cs
        var metadata = ctx.GetMetadata<int>("ScriptSequence");
        ctx.Logger.Info($"Running as script #{metadata} in sequence");

        // Read input data (from previous script)
        var data = await ctx.Csv.ReadAsync(ctx.InputPath);
        ctx.Logger.Info($"Loaded: {data.Count:N0} rows from previous step");

        // Engineer features
        var engineered = data.Select(row =>
        {
            var newRow = new Dictionary<string, string>(row);

            // Production efficiency features
            if (row.ContainsKey("생산량") && row.ContainsKey("생산필요량"))
            {
                if (TryParseDecimal(row["생산량"], out var produced) &&
                    TryParseDecimal(row["생산필요량"], out var required) &&
                    required > 0)
                {
                    var efficiency = (produced / required) * 100;
                    newRow["생산효율(%)"] = efficiency.ToString("F2");
                    newRow["생산미달"] = (produced < required) ? "1" : "0";
                }
            }

            // Inventory adequacy features
            if (row.ContainsKey("재고") && row.ContainsKey("수주량"))
            {
                if (TryParseDecimal(row["재고"], out var stock) &&
                    TryParseDecimal(row["수주량"], out var order) &&
                    order > 0)
                {
                    var coverage = (stock / order) * 100;
                    newRow["재고충분도(%)"] = coverage.ToString("F2");
                    newRow["재고부족"] = (stock < order) ? "1" : "0";
                }
            }

            // Time-based features (if datetime features exist from previous scripts)
            if (row.ContainsKey("Hour"))
            {
                if (TryParseInt(row["Hour"], out var hour))
                {
                    // Work shift classification
                    var shift = hour switch
                    {
                        >= 8 and < 16 => "1",   // Day shift
                        >= 16 and < 24 => "2",  // Evening shift
                        _ => "3"                // Night shift
                    };
                    newRow["작업교대조"] = shift;

                    // Peak/Off-peak hours
                    newRow["피크시간"] = (hour >= 9 && hour < 12) || (hour >= 14 && hour < 17) ? "1" : "0";
                }
            }

            // Quarter classification (if exists from datetime extraction)
            if (row.ContainsKey("Quarter"))
            {
                if (TryParseInt(row["Quarter"], out var quarter))
                {
                    newRow["분기"] = quarter switch
                    {
                        1 => "Q1",
                        2 => "Q2",
                        3 => "Q3",
                        4 => "Q4",
                        _ => "Unknown"
                    };
                }
            }

            return newRow;
        }).ToList();

        ctx.Logger.Info($"Engineered features for {engineered.Count:N0} rows");

        // Log feature summary
        var featureCount = engineered.FirstOrDefault()?.Count ?? 0;
        ctx.Logger.Info($"Total features: {featureCount} columns");

        // Save output
        var outputPath = Path.Combine(ctx.OutputDirectory, "03_engineered_features.csv");
        await ctx.Csv.WriteAsync(outputPath, engineered);

        ctx.Logger.Info($"✅ Saved: {outputPath}");
        return outputPath;
    }

    private bool TryParseDecimal(string value, out decimal result)
    {
        result = 0m;
        if (string.IsNullOrWhiteSpace(value))
            return false;

        // Remove thousand separators (Korean format)
        var cleaned = value.Replace(",", "").Trim();
        return decimal.TryParse(cleaned, out result);
    }

    private bool TryParseInt(string value, out int result)
    {
        result = 0;
        if (string.IsNullOrWhiteSpace(value))
            return false;

        return int.TryParse(value.Trim(), out result);
    }
}
