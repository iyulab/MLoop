using System.Text;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// Balances class distribution in training data by oversampling minority classes
/// </summary>
public class DataBalancer
{
    /// <summary>
    /// Result of data balancing operation
    /// </summary>
    public class BalanceResult
    {
        public bool Applied { get; set; }
        public string? BalancedFilePath { get; set; }
        public string? Message { get; set; }
        public int OriginalMinorityCount { get; set; }
        public int NewMinorityCount { get; set; }
        public double OriginalRatio { get; set; }
        public double NewRatio { get; set; }
    }

    /// <summary>
    /// Default target ratio for 'auto' mode (10:1)
    /// </summary>
    public const double DefaultTargetRatio = 10.0;

    /// <summary>
    /// Minimum imbalance ratio to trigger auto-balancing
    /// </summary>
    public const double AutoTriggerThreshold = 10.0;

    /// <summary>
    /// Balances the dataset by oversampling minority class
    /// </summary>
    /// <param name="dataFile">Path to input CSV file</param>
    /// <param name="labelColumn">Name of the label column</param>
    /// <param name="balanceOption">Balance option: 'none', 'auto', or target ratio (e.g., '5' for 5:1)</param>
    /// <returns>Balance result with path to balanced file if applied</returns>
    public BalanceResult Balance(string dataFile, string labelColumn, string? balanceOption)
    {
        var result = new BalanceResult { Applied = false };

        // Parse balance option
        if (string.IsNullOrWhiteSpace(balanceOption) ||
            balanceOption.Equals("none", StringComparison.OrdinalIgnoreCase))
        {
            result.Message = "Balancing disabled";
            return result;
        }

        // Read data file
        var allLines = File.ReadAllLines(dataFile, Encoding.UTF8);
        if (allLines.Length < 2)
        {
            result.Message = "No data rows found";
            return result;
        }

        var header = allLines[0];
        var columns = CsvFieldParser.ParseFields(header);
        var labelIndex = Array.IndexOf(columns, labelColumn);

        if (labelIndex == -1)
        {
            result.Message = $"Label column '{labelColumn}' not found";
            return result;
        }

        // Analyze class distribution
        var dataRows = allLines.Skip(1).ToList();
        var classCounts = new Dictionary<string, List<string>>();

        foreach (var row in dataRows)
        {
            var values = CsvFieldParser.ParseFields(row);
            if (labelIndex < values.Length)
            {
                var labelValue = values[labelIndex].Trim();
                if (!classCounts.ContainsKey(labelValue))
                {
                    classCounts[labelValue] = new List<string>();
                }
                classCounts[labelValue].Add(row);
            }
        }

        if (classCounts.Count < 2)
        {
            result.Message = "Only one class found, no balancing needed";
            return result;
        }

        // Find majority and minority classes
        var sortedClasses = classCounts
            .OrderByDescending(x => x.Value.Count)
            .ToList();

        var majorityClass = sortedClasses.First();
        var minorityClass = sortedClasses.Last();

        result.OriginalMinorityCount = minorityClass.Value.Count;
        result.OriginalRatio = (double)majorityClass.Value.Count / minorityClass.Value.Count;

        // Determine target ratio
        double targetRatio;
        if (balanceOption.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            // Auto mode: balance to DefaultTargetRatio if ratio exceeds threshold
            if (result.OriginalRatio <= AutoTriggerThreshold)
            {
                result.Message = $"Class ratio {result.OriginalRatio:F1}:1 is within acceptable range (threshold: {AutoTriggerThreshold}:1)";
                return result;
            }
            targetRatio = DefaultTargetRatio;
        }
        else if (double.TryParse(balanceOption, out var customRatio) && customRatio > 0)
        {
            targetRatio = customRatio;
        }
        else
        {
            result.Message = $"Invalid balance option: '{balanceOption}'. Use 'auto', 'none', or a number (e.g., '5' for 5:1)";
            return result;
        }

        // Already balanced enough?
        if (result.OriginalRatio <= targetRatio)
        {
            result.Message = $"Class ratio {result.OriginalRatio:F1}:1 already meets target ratio {targetRatio}:1";
            return result;
        }

        // Calculate how many minority samples needed
        var targetMinorityCount = (int)Math.Ceiling(majorityClass.Value.Count / targetRatio);
        var samplesToAdd = targetMinorityCount - minorityClass.Value.Count;

        if (samplesToAdd <= 0)
        {
            result.Message = "No oversampling needed";
            return result;
        }

        // Oversample minority class (simple random oversampling with repetition)
        var random = new Random(42); // Fixed seed for reproducibility
        var oversampledRows = new List<string>();

        for (int i = 0; i < samplesToAdd; i++)
        {
            var randomIndex = random.Next(minorityClass.Value.Count);
            oversampledRows.Add(minorityClass.Value[randomIndex]);
        }

        // Write balanced file
        var outputPath = Path.Combine(
            Path.GetDirectoryName(dataFile)!,
            Path.GetFileNameWithoutExtension(dataFile) + "_balanced.csv");

        var balancedLines = new List<string> { header };
        balancedLines.AddRange(dataRows);
        balancedLines.AddRange(oversampledRows);

        File.WriteAllLines(outputPath, balancedLines, Encoding.UTF8);

        result.Applied = true;
        result.BalancedFilePath = outputPath;
        result.NewMinorityCount = minorityClass.Value.Count + samplesToAdd;
        result.NewRatio = (double)majorityClass.Value.Count / result.NewMinorityCount;
        result.Message = $"Oversampled minority class '{minorityClass.Key}' from {minorityClass.Value.Count} to {result.NewMinorityCount} samples (ratio: {result.OriginalRatio:F1}:1 → {result.NewRatio:F1}:1)";

        return result;
    }
}
