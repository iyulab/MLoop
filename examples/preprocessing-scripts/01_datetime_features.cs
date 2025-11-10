using MLoop.Extensibility;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

/// <summary>
/// Example: DateTime parsing and feature extraction
/// Demonstrates: Converting Korean datetime strings and extracting temporal features
/// Use case: Dataset 005 (열처리 공급망최적화) with "2022.3.1 8:00" format
/// </summary>
public class DateTimeFeatureExtraction : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext ctx)
    {
        ctx.Logger.Info("=== DateTime Feature Extraction ===");

        // Read input data
        var data = await ctx.Csv.ReadAsync(ctx.InputPath);
        ctx.Logger.Info($"Loaded: {data.Count:N0} rows");

        // Process each row to extract datetime features
        var processed = data.Select(row =>
        {
            var newRow = new Dictionary<string, string>(row);

            // Parse Korean datetime format: "2022.3.1 8:00"
            if (row.ContainsKey("시간") && !string.IsNullOrEmpty(row["시간"]))
            {
                if (TryParseKoreanDateTime(row["시간"], out var dt))
                {
                    // Extract temporal features
                    newRow["Year"] = dt.Year.ToString();
                    newRow["Month"] = dt.Month.ToString();
                    newRow["Day"] = dt.Day.ToString();
                    newRow["Hour"] = dt.Hour.ToString();
                    newRow["DayOfWeek"] = ((int)dt.DayOfWeek).ToString();
                    newRow["DayOfYear"] = dt.DayOfYear.ToString();
                    newRow["WeekOfYear"] = GetWeekOfYear(dt).ToString();
                    newRow["Quarter"] = GetQuarter(dt).ToString();
                    newRow["IsWeekend"] = (dt.DayOfWeek == DayOfWeek.Saturday || dt.DayOfWeek == DayOfWeek.Sunday) ? "1" : "0";

                    ctx.Logger.Debug($"Parsed datetime: {row["시간"]} → {dt:yyyy-MM-dd HH:mm}");
                }
                else
                {
                    ctx.Logger.Warning($"Failed to parse datetime: {row["시간"]}");
                }
            }

            return newRow;
        }).ToList();

        ctx.Logger.Info($"Extracted features for {processed.Count:N0} rows");

        // Save output
        var outputPath = Path.Combine(ctx.OutputDirectory, "01_with_datetime_features.csv");
        await ctx.Csv.WriteAsync(outputPath, processed);

        ctx.Logger.Info($"✅ Saved: {outputPath}");
        return outputPath;
    }

    /// <summary>
    /// Parse Korean datetime format: "2022.3.1 8:00"
    /// </summary>
    private bool TryParseKoreanDateTime(string dateStr, out DateTime result)
    {
        result = DateTime.MinValue;

        try
        {
            // Replace Korean date separator '.' with standard '-'
            var normalized = dateStr.Replace(".", "-").Trim();

            // Try various datetime formats
            var formats = new[]
            {
                "yyyy-M-d H:mm",
                "yyyy-M-d HH:mm",
                "yyyy-MM-dd H:mm",
                "yyyy-MM-dd HH:mm"
            };

            if (DateTime.TryParseExact(normalized, formats, CultureInfo.InvariantCulture, DateTimeStyles.None, out result))
            {
                return true;
            }

            // Fallback: try general parse
            if (DateTime.TryParse(normalized, out result))
            {
                return true;
            }

            return false;
        }
        catch
        {
            return false;
        }
    }

    private int GetWeekOfYear(DateTime date)
    {
        var calendar = CultureInfo.CurrentCulture.Calendar;
        return calendar.GetWeekOfYear(date, CalendarWeekRule.FirstDay, DayOfWeek.Sunday);
    }

    private int GetQuarter(DateTime date)
    {
        return (date.Month - 1) / 3 + 1;
    }
}
