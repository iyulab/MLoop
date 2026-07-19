using Microsoft.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// The featurization-exclusion decision (DateTime / sparse / constant) is data-dependent, so it has
/// exactly one authority — <see cref="CsvDataLoader.DetermineExcludedColumns"/> — and every slice of
/// a run consumes that one decision.
/// </summary>
/// <remarks>
/// The regression these tests pin: a <em>near-constant</em> column — two distinct values across the
/// full file, the minority appearing in only a couple of rows — is genuinely constant inside one
/// train/test partition and not the other. Re-deriving the exclusion set per partition therefore
/// produces different feature widths, and the pipeline fitted on one width fails on the other with
/// "Schema mismatch for feature column 'Features': expected Vector&lt;Single, 30&gt;, got
/// Vector&lt;Single, 29&gt;". Synthetic imbalance fixtures never caught this because none of them
/// contained a column that only some partitions flatten.
/// </remarks>
public class FeatureExclusionAuthorityTests : IDisposable
{
    private readonly string _tempDirectory;
    private readonly MLContext _mlContext;
    private readonly CsvDataLoader _loader;

    public FeatureExclusionAuthorityTests()
    {
        _tempDirectory = Path.Combine(Path.GetTempPath(), $"MLoopExclusion_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempDirectory);
        _mlContext = new MLContext(seed: 42);
        _loader = new CsvDataLoader(_mlContext);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDirectory))
        {
            Directory.Delete(_tempDirectory, recursive: true);
        }
        GC.SuppressFinalize(this);
    }

    [Fact]
    public void DetermineExcludedColumns_ReportsEachReason()
    {
        var lines = new List<string> { "keep,constant_col,sparse_col,event_date,label" };
        for (int i = 0; i < 40; i++)
        {
            // sparse_col carries a value in 1 of 40 rows (97.5% missing → over the 90% threshold)
            var sparse = i == 0 ? "7" : "";
            lines.Add($"{i},SAME,{sparse},2024-01-{(i % 28) + 1:D2},{i % 2}");
        }
        var csvPath = CreateCsv("reasons.csv", lines);

        var excluded = CsvDataLoader.DetermineExcludedColumns(csvPath, "label", _ => { });

        Assert.Equal(SchemaDataTypes.ExcludedDateTime, ReasonFor(excluded, "event_date"));
        Assert.Equal(SchemaDataTypes.ExcludedSparse, ReasonFor(excluded, "sparse_col"));
        Assert.Equal(SchemaDataTypes.ExcludedConstant, ReasonFor(excluded, "constant_col"));
        Assert.DoesNotContain(excluded, c => c.Name == "keep");
        Assert.DoesNotContain(excluded, c => c.Name == "label");
    }

    [Fact]
    public void DetermineExcludedColumns_NothingToDrop_ReturnsEmpty()
    {
        var csvPath = CreateCsv("clean.csv", new[]
        {
            "a,b,label",
            "1,10,0",
            "2,20,1",
            "3,30,0"
        });

        Assert.Empty(CsvDataLoader.DetermineExcludedColumns(csvPath, "label", _ => { }));
    }

    [Fact]
    public void DetermineExcludedColumns_NearConstantColumn_DisagreesBetweenPartitions()
    {
        // Documents *why* the authority exists: the same rule, applied to two slices of one dataset,
        // legitimately reaches two different answers.
        var (trainPath, testPath) = CreateNearConstantSplit();

        var trainDecision = CsvDataLoader.DetermineExcludedColumns(trainPath, "label", _ => { });
        var testDecision = CsvDataLoader.DetermineExcludedColumns(testPath, "label", _ => { });

        Assert.Contains(trainDecision, c => c.Name == "near_constant");
        Assert.DoesNotContain(testDecision, c => c.Name == "near_constant");
    }

    [Fact]
    public void LoadData_WithSharedExclusions_GivesPartitionsTheSameSchema()
    {
        var (trainPath, testPath) = CreateNearConstantSplit();
        var fullPath = Path.Combine(_tempDirectory, "full.csv");
        File.WriteAllLines(fullPath,
            File.ReadAllLines(trainPath).Concat(File.ReadAllLines(testPath).Skip(1)));

        // One decision, taken from the full dataset, handed to both partitions.
        var exclusions = CsvDataLoader.DetermineExcludedColumns(fullPath, "label", _ => { })
            .Select(c => c.Name).ToList();

        var train = _loader.LoadData(trainPath, "label", "binary-classification", null, exclusions);
        var test = _loader.LoadData(testPath, "label", "binary-classification", null, exclusions);

        Assert.Equal(ColumnNames(train), ColumnNames(test));
        Assert.Contains("near_constant", ColumnNames(train));
    }

    [Fact]
    public void LoadData_WithoutSharedExclusions_PartitionSchemasDiverge()
    {
        // The pre-fix behaviour, kept as the negative control: each partition decides for itself and
        // the train view loses a column the test view keeps. Feeding these two into one pipeline is
        // the reported crash.
        var (trainPath, testPath) = CreateNearConstantSplit();

        var train = _loader.LoadData(trainPath, "label", "binary-classification");
        var test = _loader.LoadData(testPath, "label", "binary-classification");

        Assert.DoesNotContain("near_constant", ColumnNames(train));
        Assert.Contains("near_constant", ColumnNames(test));
    }

    [Fact]
    public void LoadData_WithEmptyExclusionSet_KeepsEveryColumn()
    {
        // An explicit "nothing is excluded" decision must be honoured, not treated as "undecided" —
        // otherwise the loader would quietly fall back to deciding for itself.
        var (trainPath, _) = CreateNearConstantSplit();

        var train = _loader.LoadData(trainPath, "label", "binary-classification", null, Array.Empty<string>());

        Assert.Contains("near_constant", ColumnNames(train));
    }

    [Fact]
    public void LoadData_NumericNearConstant_SharedExclusionsKeepFeatureWidthEqual()
    {
        // The reported crash is about the *merged* feature vector, and InferColumns only merges
        // adjacent numeric columns — a text near-constant column changes the column list without
        // changing Vector<Single, N>. This is the numeric shape, which is what actually produced
        // "expected Vector<Single, 7>, got Vector<Single, 8>" on 0.29.0.
        var (trainPath, testPath) = CreateNearConstantSplit(nearConstantValues: ("0", "1"));

        var exclusions = new List<string>(); // full data has two values → nothing excluded

        var train = _loader.LoadData(trainPath, "label", "binary-classification", null, exclusions);
        var test = _loader.LoadData(testPath, "label", "binary-classification", null, exclusions);

        Assert.Equal(FeatureWidth(train), FeatureWidth(test));
    }

    /// <summary>
    /// Total width of the numeric feature space: vector columns count for their size, scalars for one.
    /// This is the quantity ML.NET compares when it reports a "Schema mismatch for feature column".
    /// </summary>
    private static int FeatureWidth(Microsoft.ML.IDataView view) =>
        view.Schema
            .Where(c => c.Name != "label")
            .Sum(c => c.Type is Microsoft.ML.Data.VectorDataViewType vector ? vector.Size : 1);

    /// <summary>
    /// Builds a train/test pair where "near_constant" holds a single value in the train partition and
    /// two values in the test partition — the shape a stratified split produces when the off-value
    /// rows land on one side.
    /// </summary>
    private (string TrainPath, string TestPath) CreateNearConstantSplit(
        (string Common, string Rare)? nearConstantValues = null)
    {
        var (common, rare) = nearConstantValues ?? ("A", "B");
        var header = "f1,near_constant,label";
        var suffix = common + rare;

        var train = new List<string> { header };
        for (int i = 0; i < 60; i++)
            train.Add($"{i},{common},{i % 2}");

        var test = new List<string> { header };
        for (int i = 0; i < 20; i++)
            test.Add($"{i},{(i < 2 ? rare : common)},{i % 2}");

        return (CreateCsv($"train_{suffix}.csv", train), CreateCsv($"test_{suffix}.csv", test));
    }

    private static string ReasonFor(IReadOnlyList<ExcludedColumn> excluded, string name) =>
        excluded.Single(c => c.Name == name).Reason;

    private static string[] ColumnNames(Microsoft.ML.IDataView view) =>
        view.Schema.Select(c => c.Name).ToArray();

    private string CreateCsv(string fileName, IEnumerable<string> lines)
    {
        var path = Path.Combine(_tempDirectory, fileName);
        File.WriteAllLines(path, lines);
        return path;
    }
}
