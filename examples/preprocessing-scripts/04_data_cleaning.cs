using MLoop.Extensibility.Preprocessing;

/// <summary>
/// Common data cleaning operations:
/// - Remove duplicate rows
/// - Handle missing values with mean/median imputation
/// - Trim whitespace from string columns
///
/// Usage: Place in .mloop/scripts/preprocess/04_data_cleaning.cs
/// Executes automatically before training when present.
/// </summary>
public class DataCleaningScript : IPreprocessingScript
{
    public async Task<PreprocessingResult> ExecuteAsync(PreprocessingContext context)
    {
        context.Logger.Info("🧹 Data Cleaning: Removing duplicates, handling missing values");

        // Read input CSV
        var inputPath = context.InputPath;
        var outputPath = context.GetTempPath("cleaned.csv");

        try
        {
            // Use FilePrepper for efficient data cleaning
            var pipeline = FilePrepper.Pipeline.DataPipeline.LoadCsv(inputPath);

            // 1. Remove duplicate rows
            pipeline = pipeline.DropDuplicates();
            context.Logger.Info("  ✓ Removed duplicate rows");

            // 2. Fill missing numeric values with median
            // Note: This requires knowing column types. For production, you would:
            // - Detect numeric columns
            // - Apply appropriate imputation strategy per column
            // Example:
            // pipeline = pipeline.FillMissingValues(new FillMissingValuesOption
            // {
            //     Columns = new[] { "Price", "Quantity", "Amount" },
            //     Strategy = ImputationStrategy.Median
            // });

            context.Logger.Info("  ℹ️  Missing value imputation: Implement based on your data schema");

            // 3. Trim whitespace from all string columns
            // This helps with text matching and prevents issues like "Yes " != "Yes"
            var columns = await GetStringColumnsAsync(inputPath);
            if (columns.Any())
            {
                foreach (var col in columns)
                {
                    pipeline = pipeline.StringOps(col, s => s.Trim());
                }
                context.Logger.Info($"  ✓ Trimmed whitespace from {columns.Count} string column(s)");
            }

            // Save cleaned data
            pipeline.SaveCsv(outputPath);

            context.Logger.Info($"  💾 Saved cleaned data: {outputPath}");

            return new PreprocessingResult
            {
                OutputPath = outputPath,
                Success = true,
                Message = "Data cleaning completed: duplicates removed, whitespace trimmed"
            };
        }
        catch (Exception ex)
        {
            context.Logger.Error($"❌ Data cleaning failed: {ex.Message}");
            return new PreprocessingResult
            {
                OutputPath = inputPath,  // Return original on failure
                Success = false,
                Message = $"Cleaning failed: {ex.Message}"
            };
        }
    }

    private async Task<List<string>> GetStringColumnsAsync(string csvPath)
    {
        // Simple heuristic: Read first row and detect text columns
        // In production, you would use proper type inference
        var stringColumns = new List<string>();

        using var reader = new StreamReader(csvPath);
        using var csv = new CsvHelper.CsvReader(reader, new CsvHelper.Configuration.CsvConfiguration(
            System.Globalization.CultureInfo.InvariantCulture)
        {
            HasHeaderRecord = true
        });

        await csv.ReadAsync();
        csv.ReadHeader();

        if (csv.HeaderRecord != null)
        {
            await csv.ReadAsync();

            for (int i = 0; i < csv.HeaderRecord.Length; i++)
            {
                var value = csv.GetField(i);
                // If value is not numeric, consider it a string column
                if (!double.TryParse(value, out _))
                {
                    stringColumns.Add(csv.HeaderRecord[i]);
                }
            }
        }

        return stringColumns;
    }
}
