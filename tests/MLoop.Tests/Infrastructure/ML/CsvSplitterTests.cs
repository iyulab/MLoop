using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.Infrastructure.ML;

/// <summary>
/// Tests for CsvSplitter stratified CSV splitting
/// </summary>
public class CsvSplitterTests : IDisposable
{
    private readonly string _testDir;

    public CsvSplitterTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), "mloop-splitter-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_testDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDir))
        {
            try { Directory.Delete(_testDir, recursive: true); } catch { }
        }
    }

    [Fact]
    public void StratifiedSplit_PreservesClassProportions()
    {
        // Arrange - 80 class A, 20 class B
        var dataFile = CreateDataset(new Dictionary<string, int> { ["A"] = 80, ["B"] = 20 });
        var splitter = new CsvSplitter();

        // Act
        var result = splitter.StratifiedSplit(dataFile, "Label", 0.2);

        // Assert
        var trainLines = File.ReadAllLines(result.TrainFile);
        var testLines = File.ReadAllLines(result.TestFile);

        // Total rows preserved (header + data)
        Assert.Equal(100, (trainLines.Length - 1) + (testLines.Length - 1));

        // Test set has roughly 20% of each class
        var testLabels = GetLabels(testLines);
        Assert.True(testLabels.ContainsKey("A"), "Test set should contain class A");
        Assert.True(testLabels.ContainsKey("B"), "Test set should contain class B");

        // Class A: 80 total, ~16 in test
        Assert.InRange(testLabels["A"], 14, 18);
        // Class B: 20 total, ~4 in test
        Assert.InRange(testLabels["B"], 2, 6);
    }

    [Fact]
    public void StratifiedSplit_BothSetsHaveHeaders()
    {
        var dataFile = CreateDataset(new Dictionary<string, int> { ["A"] = 10, ["B"] = 10 });
        var splitter = new CsvSplitter();

        var result = splitter.StratifiedSplit(dataFile, "Label", 0.2);

        var trainHeader = File.ReadLines(result.TrainFile).First();
        var testHeader = File.ReadLines(result.TestFile).First();
        var originalHeader = File.ReadLines(dataFile).First();

        Assert.Equal(originalHeader, trainHeader);
        Assert.Equal(originalHeader, testHeader);
    }

    [Fact]
    public void StratifiedSplit_IsDeterministicWithSameSeed()
    {
        var dataFile = CreateDataset(new Dictionary<string, int> { ["A"] = 50, ["B"] = 50 });
        var splitter = new CsvSplitter();

        var result1 = splitter.StratifiedSplit(dataFile, "Label", 0.2, seed: 42);
        var lines1 = File.ReadAllLines(result1.TestFile);

        // Clean up and re-split
        File.Delete(result1.TrainFile);
        File.Delete(result1.TestFile);

        var result2 = splitter.StratifiedSplit(dataFile, "Label", 0.2, seed: 42);
        var lines2 = File.ReadAllLines(result2.TestFile);

        Assert.Equal(lines1, lines2);
    }

    [Fact]
    public void StratifiedSplit_DifferentSeedsProduceDifferentResults()
    {
        var dataFile = CreateDataset(new Dictionary<string, int> { ["A"] = 50, ["B"] = 50 });
        var splitter = new CsvSplitter();

        var result1 = splitter.StratifiedSplit(dataFile, "Label", 0.2, seed: 42);
        var lines1 = File.ReadAllLines(result1.TestFile).Skip(1).ToArray();

        File.Delete(result1.TrainFile);
        File.Delete(result1.TestFile);

        var result2 = splitter.StratifiedSplit(dataFile, "Label", 0.2, seed: 99);
        var lines2 = File.ReadAllLines(result2.TestFile).Skip(1).ToArray();

        // Different seeds should produce different splits (with high probability)
        Assert.NotEqual(lines1, lines2);
    }

    [Fact]
    public void StratifiedSplit_OutputFileNaming()
    {
        var dataFile = CreateDataset(new Dictionary<string, int> { ["A"] = 10, ["B"] = 10 }, "mydata.csv");
        var splitter = new CsvSplitter();

        var result = splitter.StratifiedSplit(dataFile, "Label", 0.2);

        Assert.EndsWith("mydata_train_split.csv", result.TrainFile);
        Assert.EndsWith("mydata_test_split.csv", result.TestFile);
    }

    [Fact]
    public void StratifiedSplit_ReportsCorrectRowCounts()
    {
        var dataFile = CreateDataset(new Dictionary<string, int> { ["A"] = 40, ["B"] = 10 });
        var splitter = new CsvSplitter();

        var result = splitter.StratifiedSplit(dataFile, "Label", 0.2);

        Assert.Equal(50, result.TrainRows + result.TestRows);
        Assert.True(result.TestRows > 0);
        Assert.True(result.TrainRows > result.TestRows);
    }

    [Fact]
    public void StratifiedSplit_EachClassHasAtLeastOneTrainRow()
    {
        // Very small minority class
        var dataFile = CreateDataset(new Dictionary<string, int> { ["A"] = 100, ["B"] = 2 });
        var splitter = new CsvSplitter();

        var result = splitter.StratifiedSplit(dataFile, "Label", 0.2);

        var trainLabels = GetLabels(File.ReadAllLines(result.TrainFile));
        Assert.True(trainLabels.ContainsKey("B"), "Train set must contain minority class");
        Assert.True(trainLabels["B"] >= 1, "Train set must have at least 1 minority sample");
    }

    [Fact]
    public void StratifiedSplit_NoDataLeakage_NoOverlapBetweenSets()
    {
        var dataFile = CreateDataset(new Dictionary<string, int> { ["A"] = 30, ["B"] = 30 });
        var splitter = new CsvSplitter();

        var result = splitter.StratifiedSplit(dataFile, "Label", 0.3);

        var trainRows = File.ReadAllLines(result.TrainFile).Skip(1).ToHashSet();
        var testRows = File.ReadAllLines(result.TestFile).Skip(1).ToHashSet();

        var overlap = trainRows.Intersect(testRows).ToList();
        // Note: identical rows can exist in both due to duplicated features, but original rows shouldn't split to both
        // With unique features per row, there should be zero overlap
        Assert.Empty(overlap);
    }

    [Fact]
    public void StratifiedSplit_MulticlassDataset()
    {
        var dataFile = CreateDataset(new Dictionary<string, int>
        {
            ["cat"] = 40,
            ["dog"] = 30,
            ["bird"] = 20,
            ["fish"] = 10
        });
        var splitter = new CsvSplitter();

        var result = splitter.StratifiedSplit(dataFile, "Label", 0.2);

        var testLabels = GetLabels(File.ReadAllLines(result.TestFile));

        // All classes should be represented in test set
        Assert.Equal(4, testLabels.Count);
        Assert.True(testLabels.ContainsKey("cat"));
        Assert.True(testLabels.ContainsKey("dog"));
        Assert.True(testLabels.ContainsKey("bird"));
        Assert.True(testLabels.ContainsKey("fish"));
    }

    [Fact]
    public void StratifiedSplit_ThrowsOnMissingLabelColumn()
    {
        var dataFile = CreateDataset(new Dictionary<string, int> { ["A"] = 10, ["B"] = 10 });
        var splitter = new CsvSplitter();

        Assert.Throws<InvalidOperationException>(() =>
            splitter.StratifiedSplit(dataFile, "NonExistent", 0.2));
    }

    [Fact]
    public void StratifiedSplit_ThrowsOnEmptyFile()
    {
        var filePath = Path.Combine(_testDir, "empty.csv");
        File.WriteAllText(filePath, "Feature1,Label\n");
        var splitter = new CsvSplitter();

        Assert.Throws<InvalidOperationException>(() =>
            splitter.StratifiedSplit(filePath, "Label", 0.2));
    }

    [Fact]
    public void StratifiedSplit_HandlesQuotedCsvFields()
    {
        var filePath = Path.Combine(_testDir, "quoted.csv");
        var lines = new List<string>
        {
            "Name,Description,Label",
            "\"Smith, John\",\"A \"\"good\"\" item\",A",
            "\"Doe, Jane\",\"Another item\",A",
            "\"Kim, Lee\",\"Third item\",A",
            "\"Park, Choi\",\"Fourth item\",B",
            "\"Han, Song\",\"Fifth item\",B",
            "\"Yoo, Kang\",\"Sixth item\",B"
        };
        File.WriteAllLines(filePath, lines);
        var splitter = new CsvSplitter();

        var result = splitter.StratifiedSplit(filePath, "Label", 0.3);

        var allTrainRows = File.ReadAllLines(result.TrainFile).Skip(1);
        var allTestRows = File.ReadAllLines(result.TestFile).Skip(1);
        Assert.Equal(6, allTrainRows.Count() + allTestRows.Count());
    }

    private string CreateDataset(Dictionary<string, int> classCounts, string fileName = "data.csv")
    {
        var filePath = Path.Combine(_testDir, fileName);
        var lines = new List<string> { "Feature1,Feature2,Label" };

        int id = 0;
        foreach (var (label, count) in classCounts)
        {
            for (int i = 0; i < count; i++)
            {
                // Use unique feature values to ensure no accidental row overlap
                lines.Add($"{id},{id * 0.1:F4},{label}");
                id++;
            }
        }

        File.WriteAllLines(filePath, lines);
        return filePath;
    }

    private Dictionary<string, int> GetLabels(string[] csvLines)
    {
        var labels = new Dictionary<string, int>();
        var header = CsvFieldParser.ParseFields(csvLines[0]);
        var labelIdx = Array.IndexOf(header, "Label");

        for (int i = 1; i < csvLines.Length; i++)
        {
            if (string.IsNullOrWhiteSpace(csvLines[i])) continue;
            var fields = CsvFieldParser.ParseFields(csvLines[i]);
            var label = fields[labelIdx].Trim();
            labels[label] = labels.GetValueOrDefault(label) + 1;
        }

        return labels;
    }
}
