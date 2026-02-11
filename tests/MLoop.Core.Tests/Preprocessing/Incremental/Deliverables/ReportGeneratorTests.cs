using Microsoft.Extensions.Logging.Abstractions;
using MLoop.Core.Preprocessing.Incremental.Deliverables;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Tests.Preprocessing.Incremental.Deliverables;

public class ReportGeneratorTests : IDisposable
{
    private readonly ReportGenerator _generator;
    private readonly string _tempDir;

    public ReportGeneratorTests()
    {
        _generator = new ReportGenerator(NullLogger<ReportGenerator>.Instance);
        _tempDir = Path.Combine(Path.GetTempPath(), $"mloop-report-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    private static IncrementalWorkflowState CreateState(
        int approvedRuleCount = 0,
        bool addStageResults = false)
    {
        var approvedRules = Enumerable.Range(0, approvedRuleCount)
            .Select(i => new PreprocessingRule
            {
                Id = $"rule-{i}",
                Type = PreprocessingRuleType.MissingValueStrategy,
                ColumnNames = new[] { $"Col{i}" },
                Description = $"Handle missing values in Col{i}",
                PatternType = PatternType.MissingValue,
                RequiresHITL = false,
                Priority = 5,
                DiscoveredInStage = 1,
                IsApproved = true,
                Confidence = 0.95
            })
            .ToList();

        var completedStages = new Dictionary<WorkflowStage, StageResult>();
        if (addStageResults)
        {
            completedStages[WorkflowStage.InitialExploration] = new StageResult
            {
                Stage = WorkflowStage.InitialExploration,
                SampleSize = 100,
                SampleRatio = 0.001,
                Analysis = new SampleAnalysis
                {
                    StageNumber = 1,
                    SampleRatio = 0.001,
                    Timestamp = DateTime.UtcNow,
                    RowCount = 100,
                    ColumnCount = 5,
                    Columns = new List<ColumnAnalysis>(),
                    QualityScore = 0.85
                },
                RulesDiscovered = approvedRules.Take(1).ToList(),
                Duration = TimeSpan.FromSeconds(5),
                CompletedAt = DateTime.UtcNow
            };
        }

        return new IncrementalWorkflowState
        {
            SessionId = "test-session-123",
            CurrentStage = WorkflowStage.Completed,
            DatasetPath = "/data/train.csv",
            TotalRecords = 10000,
            CompletedStages = completedStages,
            DiscoveredRules = new List<PreprocessingRule>(approvedRules),
            ApprovedRules = approvedRules,
            ConfidenceScore = 0.95,
            HasConverged = true,
            Config = new IncrementalWorkflowConfig()
        };
    }

    #region GenerateReport Tests

    [Fact]
    public void GenerateReport_ContainsTitle()
    {
        var state = CreateState();

        var report = _generator.GenerateReport(state);

        Assert.Contains("# Incremental Preprocessing Report", report);
    }

    [Fact]
    public void GenerateReport_ContainsSessionId()
    {
        var state = CreateState();

        var report = _generator.GenerateReport(state);

        Assert.Contains("test-session-123", report);
    }

    [Fact]
    public void GenerateReport_ContainsDatasetName()
    {
        var state = CreateState();

        var report = _generator.GenerateReport(state);

        Assert.Contains("train.csv", report);
    }

    [Fact]
    public void GenerateReport_ContainsSummary()
    {
        var state = CreateState();

        var report = _generator.GenerateReport(state);

        Assert.Contains("## Summary", report);
        Assert.Contains("Total Records", report);
        Assert.Contains("10,000", report);
        Assert.Contains("Confidence Score", report);
        Assert.Contains("Converged", report);
        Assert.Contains("Yes", report);
    }

    [Fact]
    public void GenerateReport_NoApprovedRules_ShowsEmptyMessage()
    {
        var state = CreateState(approvedRuleCount: 0);

        var report = _generator.GenerateReport(state);

        Assert.Contains("No rules were approved", report);
    }

    [Fact]
    public void GenerateReport_WithRules_ShowsRuleDetails()
    {
        var state = CreateState(approvedRuleCount: 2);

        var report = _generator.GenerateReport(state);

        Assert.Contains("Rules Applied (2 total)", report);
        Assert.Contains("Col0", report);
        Assert.Contains("Col1", report);
        Assert.Contains("MissingValueStrategy", report);
    }

    [Fact]
    public void GenerateReport_ApprovedRules_ShowCheckmark()
    {
        var state = CreateState(approvedRuleCount: 1);

        var report = _generator.GenerateReport(state);

        // Unicode checkmark for approved rules
        Assert.Contains("\u2705", report);
    }

    [Fact]
    public void GenerateReport_WithStageResults_ShowsStageDetails()
    {
        var state = CreateState(approvedRuleCount: 1, addStageResults: true);

        var report = _generator.GenerateReport(state);

        Assert.Contains("## Stage Details", report);
        Assert.Contains("InitialExploration", report);
        Assert.Contains("Sample Ratio", report);
        Assert.Contains("Quality Score", report);
    }

    [Fact]
    public void GenerateReport_ContainsDeliverablesSection()
    {
        var state = CreateState();

        var report = _generator.GenerateReport(state);

        Assert.Contains("## Deliverables", report);
        Assert.Contains("cleaned_data.csv", report);
    }

    [Fact]
    public void GenerateReport_ScriptsEnabled_ShowsScriptPath()
    {
        var state = CreateState();

        var report = _generator.GenerateReport(state);

        Assert.Contains("preprocessing_script.cs", report);
    }

    [Fact]
    public void GenerateReport_ContainsConfigurationFooter()
    {
        var state = CreateState();

        var report = _generator.GenerateReport(state);

        Assert.Contains("## Configuration", report);
        Assert.Contains("stage1Ratio", report);
        Assert.Contains("minConfidenceThreshold", report);
        Assert.Contains("```json", report);
    }

    [Fact]
    public void GenerateReport_ContainsGeneratedByFooter()
    {
        var state = CreateState();

        var report = _generator.GenerateReport(state);

        Assert.Contains("Report generated by MLoop", report);
    }

    [Fact]
    public void GenerateReport_IsValidMarkdown()
    {
        var state = CreateState(approvedRuleCount: 2, addStageResults: true);

        var report = _generator.GenerateReport(state);

        // Basic markdown structure checks
        Assert.Contains("# ", report);   // H1
        Assert.Contains("## ", report);  // H2
        Assert.Contains("### ", report); // H3
        Assert.Contains("---", report);  // Horizontal rule
        Assert.Contains("```", report);  // Code block
    }

    #endregion

    #region SaveReportAsync Tests

    [Fact]
    public async Task SaveReportAsync_CreatesFile()
    {
        var path = Path.Combine(_tempDir, "report.md");

        await _generator.SaveReportAsync("# Test Report", path);

        Assert.True(File.Exists(path));
        Assert.Equal("# Test Report", await File.ReadAllTextAsync(path));
    }

    [Fact]
    public async Task SaveReportAsync_CreatesDirectory()
    {
        var path = Path.Combine(_tempDir, "sub", "dir", "report.md");

        await _generator.SaveReportAsync("# Test", path);

        Assert.True(File.Exists(path));
    }

    #endregion
}
