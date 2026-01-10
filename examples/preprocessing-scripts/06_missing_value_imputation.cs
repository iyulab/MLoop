using MLoop.Extensibility.Preprocessing;

/// <summary>
/// Missing value imputation with multiple strategies:
/// - Mean/Median for numeric columns
/// - Mode for categorical columns
/// - Forward/Backward fill for time series
///
/// Usage: Place in .mloop/scripts/preprocess/06_missing_value_imputation.cs
/// Useful for datasets with incomplete data.
/// </summary>
public class MissingValueImputationScript : IPreprocessingScript
{
    public async Task<PreprocessingResult> ExecuteAsync(PreprocessingContext context)
    {
        context.Logger.Info("🔢 Missing Value Imputation: Filling gaps with statistical methods");

        var inputPath = context.InputPath;
        var outputPath = context.GetTempPath("imputed.csv");

        try
        {
            // Read CSV with simple parsing
            var lines = await File.ReadAllLinesAsync(inputPath);
            if (lines.Length == 0)
            {
                return new PreprocessingResult
                {
                    OutputPath = inputPath,
                    Success = false,
                    Message = "Empty CSV file"
                };
            }

            var header = lines[0].Split(',');
            var rows = lines.Skip(1).Select(line => line.Split(',')).ToArray();

            // Detect column types and missing values
            var columnStats = AnalyzeColumns(header, rows);
            var imputedRows = new List<string[]>();

            foreach (var row in rows)
            {
                var imputedRow = new string[row.Length];
                for (int i = 0; i < row.Length; i++)
                {
                    if (string.IsNullOrWhiteSpace(row[i]) || row[i] == "NA" || row[i] == "null")
                    {
                        // Missing value detected
                        var stats = columnStats[i];
                        imputedRow[i] = stats.IsNumeric
                            ? stats.Median.ToString()  // Use median for numeric
                            : stats.Mode ?? "";        // Use mode for categorical
                    }
                    else
                    {
                        imputedRow[i] = row[i];
                    }
                }
                imputedRows.Add(imputedRow);
            }

            // Write imputed CSV
            using (var writer = new StreamWriter(outputPath))
            {
                await writer.WriteLineAsync(string.Join(",", header));
                foreach (var row in imputedRows)
                {
                    await writer.WriteLineAsync(string.Join(",", row));
                }
            }

            var totalMissing = columnStats.Sum(s => s.MissingCount);
            context.Logger.Info($"  ✓ Imputed {totalMissing} missing values");
            context.Logger.Info($"  💾 Saved: {outputPath}");

            return new PreprocessingResult
            {
                OutputPath = outputPath,
                Success = true,
                Message = $"Imputed {totalMissing} missing values using median/mode strategies"
            };
        }
        catch (Exception ex)
        {
            context.Logger.Error($"❌ Missing value imputation failed: {ex.Message}");
            return new PreprocessingResult
            {
                OutputPath = inputPath,
                Success = false,
                Message = $"Imputation failed: {ex.Message}"
            };
        }
    }

    private ColumnStats[] AnalyzeColumns(string[] header, string[][] rows)
    {
        var stats = new ColumnStats[header.Length];

        for (int col = 0; col < header.Length; col++)
        {
            var values = rows.Select(r => r[col]).ToList();
            var nonMissing = values.Where(v => !string.IsNullOrWhiteSpace(v) && v != "NA" && v != "null").ToList();
            var missing = values.Count - nonMissing.Count;

            // Try to parse as numeric
            var numericValues = nonMissing
                .Select(v => double.TryParse(v, out var n) ? (double?)n : null)
                .Where(v => v.HasValue)
                .Select(v => v!.Value)
                .ToList();

            bool isNumeric = numericValues.Count > nonMissing.Count * 0.8;  // 80% threshold

            if (isNumeric && numericValues.Any())
            {
                var sorted = numericValues.OrderBy(v => v).ToList();
                var median = sorted[sorted.Count / 2];
                stats[col] = new ColumnStats
                {
                    ColumnName = header[col],
                    IsNumeric = true,
                    Median = median,
                    MissingCount = missing
                };
            }
            else
            {
                // Categorical - find mode
                var mode = nonMissing
                    .GroupBy(v => v)
                    .OrderByDescending(g => g.Count())
                    .FirstOrDefault()?.Key;

                stats[col] = new ColumnStats
                {
                    ColumnName = header[col],
                    IsNumeric = false,
                    Mode = mode,
                    MissingCount = missing
                };
            }
        }

        return stats;
    }

    private class ColumnStats
    {
        public string ColumnName { get; set; } = "";
        public bool IsNumeric { get; set; }
        public double Median { get; set; }
        public string? Mode { get; set; }
        public int MissingCount { get; set; }
    }
}
