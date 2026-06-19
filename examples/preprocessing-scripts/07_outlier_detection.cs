using MLoop.Extensibility.Preprocessing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

/// <summary>
/// Outlier detection and handling using the IQR (Interquartile Range) method:
/// - Detect outliers: values &lt; Q1 - 1.5*IQR or &gt; Q3 + 1.5*IQR
/// - This example removes rows containing any outlier value
///
/// Usage: Place in .mloop/scripts/preprocess/07_outlier_detection.cs
/// Useful for cleaning datasets with extreme values.
///
/// Contract: IPreprocessingScript.ExecuteAsync returns the path to the produced CSV.
/// </summary>
public class OutlierDetectionScript : IPreprocessingScript
{
    private const double IQR_MULTIPLIER = 1.5;  // Standard IQR multiplier for outlier detection

    public async Task<string> ExecuteAsync(PreprocessContext ctx)
    {
        ctx.Logger.Info("📊 Outlier Detection: Identifying extreme values using IQR method");

        var inputPath = ctx.InputPath;

        // Read CSV.
        var lines = await File.ReadAllLinesAsync(inputPath);
        if (lines.Length == 0)
        {
            ctx.Logger.Warning("  ⚠️ Empty CSV file, passing through unchanged");
            return inputPath;
        }

        var header = lines[0].Split(',');
        var rows = lines.Skip(1).Select(line => line.Split(',')).ToArray();

        // Detect numeric columns and calculate outlier bounds.
        var columnBounds = CalculateOutlierBounds(header, rows);

        // Filter rows with outliers.
        var cleanRows = new List<string[]>();
        var outlierCount = 0;

        foreach (var row in rows)
        {
            bool hasOutlier = false;

            for (int i = 0; i < row.Length; i++)
            {
                if (columnBounds[i] == null) continue;  // Skip non-numeric columns

                if (double.TryParse(row[i], out var value))
                {
                    var bounds = columnBounds[i]!;
                    if (value < bounds.LowerBound || value > bounds.UpperBound)
                    {
                        hasOutlier = true;
                        break;
                    }
                }
            }

            if (!hasOutlier)
            {
                cleanRows.Add(row);
            }
            else
            {
                outlierCount++;
            }
        }

        // Write cleaned CSV.
        var outputPath = Path.Combine(ctx.OutputDirectory, "07_outliers_removed.csv");
        using (var writer = new StreamWriter(outputPath))
        {
            await writer.WriteLineAsync(string.Join(",", header));
            foreach (var row in cleanRows)
            {
                await writer.WriteLineAsync(string.Join(",", row));
            }
        }

        var percentRemoved = rows.Length == 0 ? 0 : (outlierCount * 100.0) / rows.Length;
        ctx.Logger.Info($"  ✓ Removed {outlierCount} rows with outliers ({percentRemoved:F1}%)");
        ctx.Logger.Info($"  ✓ Retained {cleanRows.Count} clean rows");

        // Log outlier bounds for each numeric column.
        for (int i = 0; i < header.Length; i++)
        {
            if (columnBounds[i] != null)
            {
                var bounds = columnBounds[i]!;
                ctx.Logger.Info($"  📏 {header[i]}: [{bounds.LowerBound:F2}, {bounds.UpperBound:F2}]");
            }
        }

        ctx.Logger.Info($"✅ Saved: {outputPath}");
        return outputPath;
    }

    private OutlierBounds?[] CalculateOutlierBounds(string[] header, string[][] rows)
    {
        var bounds = new OutlierBounds?[header.Length];

        for (int col = 0; col < header.Length; col++)
        {
            var values = rows
                .Select(r => r[col])
                .Where(v => !string.IsNullOrWhiteSpace(v))
                .Select(v => double.TryParse(v, out var n) ? (double?)n : null)
                .Where(v => v.HasValue)
                .Select(v => v!.Value)
                .OrderBy(v => v)
                .ToList();

            // Need at least 4 values to calculate quartiles.
            if (values.Count < 4)
            {
                bounds[col] = null;
                continue;
            }

            // Calculate Q1, Q3, and IQR.
            var q1Index = values.Count / 4;
            var q3Index = (values.Count * 3) / 4;
            var q1 = values[q1Index];
            var q3 = values[q3Index];
            var iqr = q3 - q1;

            bounds[col] = new OutlierBounds
            {
                LowerBound = q1 - (IQR_MULTIPLIER * iqr),
                UpperBound = q3 + (IQR_MULTIPLIER * iqr),
                Q1 = q1,
                Q3 = q3,
                IQR = iqr
            };
        }

        return bounds;
    }

    private class OutlierBounds
    {
        public double LowerBound { get; set; }
        public double UpperBound { get; set; }
        public double Q1 { get; set; }
        public double Q3 { get; set; }
        public double IQR { get; set; }
    }
}
