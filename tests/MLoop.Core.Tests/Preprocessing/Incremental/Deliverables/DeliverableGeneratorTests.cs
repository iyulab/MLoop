using System.Text.Json;
using Microsoft.Data.Analysis;
using Microsoft.Extensions.Logging.Abstractions;
using MLoop.Core.Preprocessing.Incremental.Deliverables;
using MLoop.Core.Preprocessing.Incremental.Deliverables.Contracts;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;
using MLoop.Core.Preprocessing.Incremental.ScriptGeneration;
using MLoop.Core.Preprocessing.Incremental.ScriptGeneration.Contracts;
using MLoop.Core.Preprocessing.Incremental.ScriptGeneration.Models;

namespace MLoop.Core.Tests.Preprocessing.Incremental.Deliverables;

public class DeliverableGeneratorTests : IDisposable
{
    private readonly DeliverableGenerator _generator;
    private readonly string _tempDir;

    public DeliverableGeneratorTests()
    {
        var scriptGenerator = new ScriptGenerator(NullLogger<ScriptGenerator>.Instance);
        var reportGenerator = new ReportGenerator(NullLogger<ReportGenerator>.Instance);
        _generator = new DeliverableGenerator(
            scriptGenerator,
            reportGenerator,
            NullLogger<DeliverableGenerator>.Instance);
        _tempDir = Path.Combine(Path.GetTempPath(), $"mloop-deliv-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    private static DataFrame CreateTestData()
    {
        var df = new DataFrame();
        df.Columns.Add(new PrimitiveDataFrameColumn<double>("Value", new[] { 1.0, 2.0, 3.0 }));
        df.Columns.Add(new StringDataFrameColumn("Name", new[] { "A", "B", "C" }));
        return df;
    }

    private static IncrementalWorkflowState CreateState(
        bool generateScripts = true,
        bool generateReport = true,
        int approvedRuleCount = 1)
    {
        var rules = Enumerable.Range(0, approvedRuleCount)
            .Select(i => new PreprocessingRule
            {
                Id = $"rule-{i}",
                Type = PreprocessingRuleType.MissingValueStrategy,
                ColumnNames = new[] { "Value" },
                Description = $"Handle missing values {i}",
                PatternType = PatternType.MissingValue,
                RequiresHITL = false,
                Priority = 5,
                DiscoveredInStage = 1,
                IsApproved = true,
                Confidence = 0.95
            }).ToList();

        return new IncrementalWorkflowState
        {
            SessionId = "test-session",
            CurrentStage = WorkflowStage.Completed,
            DatasetPath = "/data/test.csv",
            TotalRecords = 1000,
            CompletedStages = new Dictionary<WorkflowStage, StageResult>(),
            DiscoveredRules = new List<PreprocessingRule>(rules),
            ApprovedRules = rules,
            ConfidenceScore = 0.95,
            Config = new IncrementalWorkflowConfig
            {
                GenerateScripts = generateScripts,
                GenerateReport = generateReport
            }
        };
    }

    #region GenerateAllAsync Tests

    [Fact]
    public async Task GenerateAllAsync_CreatesCleanedDataFile()
    {
        var outputDir = Path.Combine(_tempDir, "output1");

        var manifest = await _generator.GenerateAllAsync(
            CreateState(), CreateTestData(), outputDir);

        Assert.True(File.Exists(manifest.CleanedDataPath));
    }

    [Fact]
    public async Task GenerateAllAsync_ScriptsEnabled_CreatesScriptFile()
    {
        var outputDir = Path.Combine(_tempDir, "output2");

        var manifest = await _generator.GenerateAllAsync(
            CreateState(generateScripts: true), CreateTestData(), outputDir);

        Assert.NotNull(manifest.ScriptPath);
        Assert.True(File.Exists(manifest.ScriptPath));
    }

    [Fact]
    public async Task GenerateAllAsync_ScriptsDisabled_NoScriptFile()
    {
        var outputDir = Path.Combine(_tempDir, "output3");

        var manifest = await _generator.GenerateAllAsync(
            CreateState(generateScripts: false), CreateTestData(), outputDir);

        Assert.Null(manifest.ScriptPath);
    }

    [Fact]
    public async Task GenerateAllAsync_NoApprovedRules_NoScriptFile()
    {
        var outputDir = Path.Combine(_tempDir, "output4");

        var manifest = await _generator.GenerateAllAsync(
            CreateState(generateScripts: true, approvedRuleCount: 0), CreateTestData(), outputDir);

        Assert.Null(manifest.ScriptPath);
    }

    [Fact]
    public async Task GenerateAllAsync_ReportEnabled_CreatesReportFile()
    {
        var outputDir = Path.Combine(_tempDir, "output5");

        var manifest = await _generator.GenerateAllAsync(
            CreateState(generateReport: true), CreateTestData(), outputDir);

        Assert.NotNull(manifest.ReportPath);
        Assert.True(File.Exists(manifest.ReportPath));
    }

    [Fact]
    public async Task GenerateAllAsync_ReportDisabled_NoReportFile()
    {
        var outputDir = Path.Combine(_tempDir, "output6");

        var manifest = await _generator.GenerateAllAsync(
            CreateState(generateReport: false), CreateTestData(), outputDir);

        Assert.Null(manifest.ReportPath);
    }

    [Fact]
    public async Task GenerateAllAsync_AlwaysCreatesMetadata()
    {
        var outputDir = Path.Combine(_tempDir, "output7");

        var manifest = await _generator.GenerateAllAsync(
            CreateState(), CreateTestData(), outputDir);

        Assert.NotNull(manifest.MetadataPath);
        Assert.True(File.Exists(manifest.MetadataPath));
    }

    [Fact]
    public async Task GenerateAllAsync_CreatesOutputDirectory()
    {
        var outputDir = Path.Combine(_tempDir, "new", "dir");

        await _generator.GenerateAllAsync(
            CreateState(), CreateTestData(), outputDir);

        Assert.True(Directory.Exists(outputDir));
    }

    #endregion

    #region SaveCleanedDataAsync Tests

    [Fact]
    public async Task SaveCleanedDataAsync_CreatesCsvFile()
    {
        var path = Path.Combine(_tempDir, "data.csv");
        var df = CreateTestData();

        await _generator.SaveCleanedDataAsync(df, path);

        Assert.True(File.Exists(path));
        var content = await File.ReadAllTextAsync(path);
        Assert.Contains("Value", content);
        Assert.Contains("Name", content);
    }

    #endregion

    #region GenerateMetadataAsync Tests

    [Fact]
    public async Task GenerateMetadataAsync_CreatesValidJson()
    {
        var path = Path.Combine(_tempDir, "meta.json");
        var state = CreateState();

        await _generator.GenerateMetadataAsync(state, path);

        var json = await File.ReadAllTextAsync(path);
        var doc = JsonDocument.Parse(json);
        Assert.Equal("test-session", doc.RootElement.GetProperty("sessionId").GetString());
    }

    [Fact]
    public async Task GenerateMetadataAsync_IncludesConfig()
    {
        var path = Path.Combine(_tempDir, "meta2.json");
        var state = CreateState();

        await _generator.GenerateMetadataAsync(state, path);

        var json = await File.ReadAllTextAsync(path);
        var doc = JsonDocument.Parse(json);
        Assert.True(doc.RootElement.TryGetProperty("config", out _));
    }

    [Fact]
    public async Task GenerateMetadataAsync_IncludesRules()
    {
        var path = Path.Combine(_tempDir, "meta3.json");
        var state = CreateState(approvedRuleCount: 2);

        await _generator.GenerateMetadataAsync(state, path);

        var json = await File.ReadAllTextAsync(path);
        var doc = JsonDocument.Parse(json);
        var rules = doc.RootElement.GetProperty("rulesDiscovered");
        Assert.Equal(2, rules.GetArrayLength());
    }

    #endregion
}
