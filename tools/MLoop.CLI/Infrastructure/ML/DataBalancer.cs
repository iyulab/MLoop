using System.Text;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.Core.Data;
using MLoop.Core.Prediction;
using MLoop.Core.Training;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// Balances class distribution in a CSV training file by oversampling minority classes — the rule is
/// <see cref="ClassBalance"/>; this reads the rows and writes the balanced file.
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

        // Flatten multiline quoted fields before line-by-line processing
        dataFile = CsvDataLoader.FlattenMultiLineQuotedFields(dataFile, CoreNarration.Sink);

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

        var outcome = ClassBalance.Oversample(
            classCounts.Select(c => KeyValuePair.Create(c.Key, (IReadOnlyList<string>)c.Value)).ToList(), balanceOption);
        result.Message = outcome.Message;
        result.OriginalMinorityCount = outcome.OriginalMinorityCount;
        result.OriginalRatio = outcome.OriginalRatio;
        if (!outcome.Applied)
            return result;

        // Write balanced file
        var outputPath = Path.Combine(
            Path.GetDirectoryName(dataFile)!,
            Path.GetFileNameWithoutExtension(dataFile) + "_balanced.csv");

        var balancedLines = new List<string> { header };
        balancedLines.AddRange(dataRows);
        balancedLines.AddRange(outcome.Copies);

        File.WriteAllLines(outputPath, balancedLines, Encoding.UTF8);

        result.Applied = true;
        result.BalancedFilePath = outputPath;
        result.NewMinorityCount = outcome.NewMinorityCount;
        result.NewRatio = outcome.NewRatio;

        return result;
    }
}
