using System.Text;
using Microsoft.Extensions.Logging;
using MLoop.Core.Preprocessing.Incremental.Models;

namespace MLoop.Core.Preprocessing.Incremental.Deliverables;

/// <summary>
/// Generates markdown processing reports from workflow state.
/// </summary>
public sealed class ReportGenerator
{
    private readonly ILogger<ReportGenerator> _logger;

    public ReportGenerator(ILogger<ReportGenerator> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Generates a comprehensive markdown report from workflow state.
    /// </summary>
    public string GenerateReport(IncrementalWorkflowState state)
    {
        var sb = new StringBuilder();

        // Title and overview
        GenerateTitle(sb, state);
        GenerateSummary(sb, state);

        // Rules section
        GenerateRulesSection(sb, state);

        // Stage details
        GenerateStageDetails(sb, state);

        // Deliverables section
        GenerateDeliverablesSection(sb, state);

        // Footer
        GenerateFooter(sb, state);

        var report = sb.ToString();

        _logger.LogInformation("Generated report with {LineCount} lines", report.Split('\n').Length);

        return report;
    }

    /// <summary>
    /// Saves the report to a file.
    /// </summary>
    public async Task SaveReportAsync(
        string report,
        string outputPath,
        CancellationToken cancellationToken = default)
    {
        var directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        await File.WriteAllTextAsync(outputPath, report, cancellationToken);

        _logger.LogInformation("Report saved to: {Path}", outputPath);
    }

    // ===== Report Generation Methods =====

    private void GenerateTitle(StringBuilder sb, IncrementalWorkflowState state)
    {
        sb.AppendLine("# Incremental Preprocessing Report");
        sb.AppendLine();
        sb.AppendLine($"**Session ID**: `{state.SessionId}`");
        sb.AppendLine($"**Dataset**: `{Path.GetFileName(state.DatasetPath)}`");
        sb.AppendLine($"**Generated**: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");
        sb.AppendLine();
        sb.AppendLine("---");
        sb.AppendLine();
    }

    private void GenerateSummary(StringBuilder sb, IncrementalWorkflowState state)
    {
        sb.AppendLine("## Summary");
        sb.AppendLine();
        sb.AppendLine($"- **Total Records**: {state.TotalRecords:N0}");
        sb.AppendLine($"- **Processing Duration**: {state.TotalDuration:hh\\:mm\\:ss}");
        sb.AppendLine($"- **Workflow Stage**: {state.CurrentStage}");
        sb.AppendLine($"- **Confidence Score**: {state.ConfidenceScore:P2}");
        sb.AppendLine($"- **Converged**: {(state.HasConverged ? "Yes" : "No")}");
        sb.AppendLine();

        var completedStages = state.CompletedStages.Count;
        var totalStages = Enum.GetValues<WorkflowStage>().Length - 2; // Exclude NotStarted and Completed

        sb.AppendLine($"- **Stages Completed**: {completedStages}/{totalStages}");
        sb.AppendLine($"- **Rules Discovered**: {state.DiscoveredRules.Count}");
        sb.AppendLine($"- **Rules Approved**: {state.ApprovedRules.Count}");
        sb.AppendLine();
        sb.AppendLine("---");
        sb.AppendLine();
    }

    private void GenerateRulesSection(StringBuilder sb, IncrementalWorkflowState state)
    {
        sb.AppendLine($"## Rules Applied ({state.ApprovedRules.Count} total)");
        sb.AppendLine();

        if (state.ApprovedRules.Count == 0)
        {
            sb.AppendLine("*No rules were approved for application.*");
            sb.AppendLine();
            return;
        }

        for (var i = 0; i < state.ApprovedRules.Count; i++)
        {
            var rule = state.ApprovedRules[i];
            var checkmark = rule.IsApproved ? "✅" : "❌";

            sb.AppendLine($"{i + 1}. {checkmark} **{rule.Type}**");
            sb.AppendLine($"   - **Columns**: {string.Join(", ", rule.ColumnNames)}");
            sb.AppendLine($"   - **Description**: {rule.Description}");
            sb.AppendLine($"   - **Confidence**: {rule.Confidence:P2}");

            if (!string.IsNullOrEmpty(rule.UserFeedback))
            {
                sb.AppendLine($"   - **User Feedback**: {rule.UserFeedback}");
            }

            sb.AppendLine();
        }

        sb.AppendLine("---");
        sb.AppendLine();
    }

    private void GenerateStageDetails(StringBuilder sb, IncrementalWorkflowState state)
    {
        sb.AppendLine("## Stage Details");
        sb.AppendLine();

        foreach (var stage in Enum.GetValues<WorkflowStage>())
        {
            if (stage == WorkflowStage.NotStarted || stage == WorkflowStage.Completed)
            {
                continue;
            }

            if (!state.CompletedStages.TryGetValue(stage, out var stageResult))
            {
                continue;
            }

            sb.AppendLine($"### {stage}");
            sb.AppendLine();
            sb.AppendLine($"- **Sample Ratio**: {stageResult.SampleRatio:P2}");
            sb.AppendLine($"- **Sample Size**: {stageResult.SampleSize:N0} records");
            sb.AppendLine($"- **Duration**: {stageResult.Duration:hh\\:mm\\:ss}");
            sb.AppendLine($"- **Rules Discovered**: {stageResult.RulesDiscovered.Count}");
            sb.AppendLine($"- **Quality Score**: {stageResult.Analysis.QualityScore:P2}");

            if (!string.IsNullOrEmpty(stageResult.Notes))
            {
                sb.AppendLine($"- **Notes**: {stageResult.Notes}");
            }

            sb.AppendLine();
        }

        sb.AppendLine("---");
        sb.AppendLine();
    }

    private void GenerateDeliverablesSection(StringBuilder sb, IncrementalWorkflowState state)
    {
        sb.AppendLine("## Deliverables");
        sb.AppendLine();

        var outputDir = state.Config.OutputDirectory;

        sb.AppendLine($"- **Cleaned Data**: `{outputDir}/cleaned_data.csv`");

        if (state.Config.GenerateScripts)
        {
            sb.AppendLine($"- **Preprocessing Script**: `{outputDir}/preprocessing_script.cs`");
        }

        if (state.Config.GenerateReport)
        {
            sb.AppendLine($"- **This Report**: `{outputDir}/report.md`");
        }

        sb.AppendLine($"- **Workflow Metadata**: `{outputDir}/metadata.json`");
        sb.AppendLine();
        sb.AppendLine("---");
        sb.AppendLine();
    }

    private void GenerateFooter(StringBuilder sb, IncrementalWorkflowState state)
    {
        sb.AppendLine("## Configuration");
        sb.AppendLine();
        sb.AppendLine("```json");
        sb.AppendLine("{");
        sb.AppendLine($"  \"stage1Ratio\": {state.Config.Stage1Ratio},");
        sb.AppendLine($"  \"stage2Ratio\": {state.Config.Stage2Ratio},");
        sb.AppendLine($"  \"stage3Ratio\": {state.Config.Stage3Ratio},");
        sb.AppendLine($"  \"stage4Ratio\": {state.Config.Stage4Ratio},");
        sb.AppendLine($"  \"minConfidenceThreshold\": {state.Config.MinConfidenceThreshold},");
        sb.AppendLine($"  \"maxErrorRate\": {state.Config.MaxErrorRate},");
        sb.AppendLine($"  \"skipHITL\": {state.Config.SkipHITL.ToString().ToLower()},");
        sb.AppendLine($"  \"enableAutoApproval\": {state.Config.EnableAutoApproval.ToString().ToLower()}");
        sb.AppendLine("}");
        sb.AppendLine("```");
        sb.AppendLine();
        sb.AppendLine("---");
        sb.AppendLine();
        sb.AppendLine("*Report generated by MLoop Incremental Preprocessing*");
    }
}
