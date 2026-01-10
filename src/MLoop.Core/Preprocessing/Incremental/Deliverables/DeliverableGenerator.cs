using System.Text.Json;
using Microsoft.Data.Analysis;
using Microsoft.Extensions.Logging;
using MLoop.Core.Preprocessing.Incremental.Deliverables.Contracts;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.ScriptGeneration.Contracts;

namespace MLoop.Core.Preprocessing.Incremental.Deliverables;

/// <summary>
/// Generates all deliverables from a completed preprocessing workflow.
/// </summary>
public sealed class DeliverableGenerator : IDeliverableGenerator
{
    private readonly IScriptGenerator _scriptGenerator;
    private readonly ReportGenerator _reportGenerator;
    private readonly ILogger<DeliverableGenerator> _logger;

    public DeliverableGenerator(
        IScriptGenerator scriptGenerator,
        ReportGenerator reportGenerator,
        ILogger<DeliverableGenerator> logger)
    {
        _scriptGenerator = scriptGenerator ?? throw new ArgumentNullException(nameof(scriptGenerator));
        _reportGenerator = reportGenerator ?? throw new ArgumentNullException(nameof(reportGenerator));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <inheritdoc />
    public async Task<DeliverableManifest> GenerateAllAsync(
        IncrementalWorkflowState state,
        DataFrame cleanedData,
        string outputDirectory,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Generating all deliverables to: {OutputDirectory}", outputDirectory);

        // Ensure output directory exists
        if (!Directory.Exists(outputDirectory))
        {
            Directory.CreateDirectory(outputDirectory);
        }

        var manifest = new DeliverableManifest
        {
            CleanedDataPath = Path.Combine(outputDirectory, "cleaned_data.csv")
        };

        // 1. Save cleaned data
        await SaveCleanedDataAsync(cleanedData, manifest.CleanedDataPath, cancellationToken);

        string? scriptPath = null;
        string? reportPath = null;
        string? metadataPath;

        // 2. Generate preprocessing script (if enabled)
        if (state.Config.GenerateScripts && state.ApprovedRules.Count > 0)
        {
            scriptPath = Path.Combine(outputDirectory, "preprocessing_script.cs");
            await _scriptGenerator.GenerateAndSaveAsync(
                state.ApprovedRules,
                scriptPath,
                null,
                cancellationToken);
        }

        // 3. Generate processing report (if enabled)
        if (state.Config.GenerateReport)
        {
            reportPath = Path.Combine(outputDirectory, "report.md");
            await GenerateReportAsync(state, reportPath, cancellationToken);
        }

        // 4. Generate workflow metadata
        metadataPath = Path.Combine(outputDirectory, "metadata.json");
        await GenerateMetadataAsync(state, metadataPath, cancellationToken);

        // Create final manifest
        manifest = new DeliverableManifest
        {
            CleanedDataPath = manifest.CleanedDataPath,
            ScriptPath = scriptPath,
            ReportPath = reportPath,
            MetadataPath = metadataPath
        };

        _logger.LogInformation("All deliverables generated successfully");

        return manifest;
    }

    /// <inheritdoc />
    public async Task SaveCleanedDataAsync(
        DataFrame data,
        string outputPath,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Saving cleaned data to: {Path}", outputPath);

        var directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        // Save as CSV using DataFrame.SaveCsv
        DataFrame.SaveCsv(data, outputPath);

        _logger.LogInformation("Cleaned data saved. Rows: {RowCount}", data.Rows.Count);

        await Task.CompletedTask; // For async consistency
    }

    /// <inheritdoc />
    public async Task GenerateReportAsync(
        IncrementalWorkflowState state,
        string outputPath,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Generating processing report to: {Path}", outputPath);

        var report = _reportGenerator.GenerateReport(state);
        await _reportGenerator.SaveReportAsync(report, outputPath, cancellationToken);

        _logger.LogInformation("Report generated successfully");
    }

    /// <inheritdoc />
    public async Task GenerateMetadataAsync(
        IncrementalWorkflowState state,
        string outputPath,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Generating workflow metadata to: {Path}", outputPath);

        var directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var metadata = new
        {
            sessionId = state.SessionId,
            currentStage = state.CurrentStage.ToString(),
            datasetPath = state.DatasetPath,
            totalRecords = state.TotalRecords,
            totalDuration = state.TotalDuration.ToString(@"hh\:mm\:ss"),
            confidenceScore = state.ConfidenceScore,
            hasConverged = state.HasConverged,
            startedAt = state.StartedAt,
            completedAt = state.CompletedAt,
            completedStages = state.CompletedStages.ToDictionary(
                kvp => kvp.Key.ToString(),
                kvp => new
                {
                    sampleSize = kvp.Value.SampleSize,
                    sampleRatio = kvp.Value.SampleRatio,
                    rulesDiscovered = kvp.Value.RulesDiscovered.Count,
                    duration = kvp.Value.Duration.ToString(@"hh\:mm\:ss"),
                    qualityScore = kvp.Value.Analysis.QualityScore
                }),
            rulesDiscovered = state.DiscoveredRules.Select(r => new
            {
                id = r.Id,
                type = r.Type.ToString(),
                columnNames = r.ColumnNames,
                description = r.Description,
                confidence = r.Confidence,
                isApproved = r.IsApproved,
                userFeedback = r.UserFeedback
            }).ToList(),
            config = new
            {
                stage1Ratio = state.Config.Stage1Ratio,
                stage2Ratio = state.Config.Stage2Ratio,
                stage3Ratio = state.Config.Stage3Ratio,
                stage4Ratio = state.Config.Stage4Ratio,
                minConfidenceThreshold = state.Config.MinConfidenceThreshold,
                maxErrorRate = state.Config.MaxErrorRate,
                skipHITL = state.Config.SkipHITL,
                enableAutoApproval = state.Config.EnableAutoApproval,
                generateScripts = state.Config.GenerateScripts,
                generateReport = state.Config.GenerateReport
            }
        };

        var json = JsonSerializer.Serialize(metadata, new JsonSerializerOptions
        {
            WriteIndented = true
        });

        await File.WriteAllTextAsync(outputPath, json, cancellationToken);

        _logger.LogInformation("Metadata generated successfully");
    }
}
