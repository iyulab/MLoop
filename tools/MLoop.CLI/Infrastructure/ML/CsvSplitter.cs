using System.Text;
using MLoop.Core.Data;
using MLoop.Core.Prediction;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// Stratified CSV splitter that preserves class distribution in train/test sets.
/// Used to split data BEFORE balancing to prevent data leakage.
/// </summary>
public class CsvSplitter
{
    /// <summary>
    /// Result of a stratified split operation
    /// </summary>
    public class SplitResult
    {
        public required string TrainFile { get; init; }
        public required string TestFile { get; init; }
        public int TrainRows { get; init; }
        public int TestRows { get; init; }

        /// <summary>
        /// Per-class train/test counts. The point of stratifying is that every class survives into
        /// both partitions, so the split reports what it actually achieved rather than leaving the
        /// caller to re-scan the files to find out.
        /// </summary>
        public IReadOnlyDictionary<string, (int Train, int Test)> PerClass { get; init; }
            = new Dictionary<string, (int, int)>();
    }

    /// <summary>
    /// Splits a CSV file into train and test sets using stratified sampling.
    /// Each class in the label column maintains its proportion in both sets.
    /// </summary>
    /// <param name="dataFile">Path to input CSV file</param>
    /// <param name="labelColumn">Name of the label column for stratification</param>
    /// <param name="testFraction">Fraction of data for test set (0.0-1.0)</param>
    /// <param name="seed">Random seed for reproducibility</param>
    /// <param name="outputDirectory">
    /// Where to write the two split files. Defaults to the input file's directory, which is what the
    /// <c>--balance</c> flow wants (the splits are user-visible artifacts there). The default split
    /// path passes a temp directory instead, so routine training does not litter <c>datasets/</c>.
    /// </param>
    /// <returns>Paths to train and test split files</returns>
    public SplitResult StratifiedSplit(string dataFile, string labelColumn, double testFraction, int seed = 42,
        string? outputDirectory = null)
    {
        // Flatten multiline quoted fields before line-by-line processing
        dataFile = CsvDataLoader.FlattenMultiLineQuotedFields(dataFile);

        var allLines = File.ReadAllLines(dataFile, Encoding.UTF8);
        if (allLines.Length < 2)
        {
            throw new InvalidOperationException("CSV file has no data rows");
        }

        var header = allLines[0];
        var columns = CsvFieldParser.ParseFields(header);
        var labelIndex = Array.IndexOf(columns, labelColumn);

        if (labelIndex == -1)
        {
            throw new InvalidOperationException($"Label column '{labelColumn}' not found in CSV header");
        }

        // Group rows by class label
        var classBuckets = new Dictionary<string, List<string>>();
        for (int i = 1; i < allLines.Length; i++)
        {
            var row = allLines[i];
            if (string.IsNullOrWhiteSpace(row)) continue;

            var fields = CsvFieldParser.ParseFields(row);
            var label = labelIndex < fields.Length ? fields[labelIndex].Trim() : "";

            if (!classBuckets.TryGetValue(label, out var bucket))
            {
                bucket = new List<string>();
                classBuckets[label] = bucket;
            }
            bucket.Add(row);
        }

        // Stratified split: take testFraction from each class
        var random = new Random(seed);
        var trainRows = new List<string>();
        var testRows = new List<string>();
        var perClass = new Dictionary<string, (int Train, int Test)>();

        foreach (var (label, rows) in classBuckets)
        {
            // Shuffle within each class
            var shuffled = rows.OrderBy(_ => random.Next()).ToList();

            var testCount = Math.Max(1, (int)Math.Round(shuffled.Count * testFraction));
            // Ensure at least 1 train row per class
            if (testCount >= shuffled.Count)
            {
                testCount = shuffled.Count - 1;
            }

            testRows.AddRange(shuffled.Take(testCount));
            trainRows.AddRange(shuffled.Skip(testCount));
            perClass[label] = (shuffled.Count - testCount, testCount);
        }

        // Write output files
        var baseName = Path.GetFileNameWithoutExtension(dataFile);
        var dir = outputDirectory ?? Path.GetDirectoryName(dataFile)!;
        Directory.CreateDirectory(dir);

        var trainFile = Path.Combine(dir, $"{baseName}_train_split.csv");
        var testFile = Path.Combine(dir, $"{baseName}_test_split.csv");

        var trainLines = new List<string>(trainRows.Count + 1) { header };
        trainLines.AddRange(trainRows);
        File.WriteAllLines(trainFile, trainLines, Encoding.UTF8);

        var testLines = new List<string>(testRows.Count + 1) { header };
        testLines.AddRange(testRows);
        File.WriteAllLines(testFile, testLines, Encoding.UTF8);

        return new SplitResult
        {
            TrainFile = trainFile,
            TestFile = testFile,
            TrainRows = trainRows.Count,
            TestRows = testRows.Count,
            PerClass = perClass
        };
    }
}
