using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.Models;

namespace MLoop.Core.Preprocessing.Incremental.Deliverables.Contracts;

/// <summary>
/// Generates all deliverables from a completed preprocessing workflow.
/// Includes cleaned data, scripts, reports, and metadata.
/// </summary>
public interface IDeliverableGenerator
{
    /// <summary>
    /// Generates all deliverables from a completed workflow.
    /// </summary>
    /// <param name="state">The completed workflow state.</param>
    /// <param name="cleanedData">The preprocessed DataFrame.</param>
    /// <param name="outputDirectory">Directory to save deliverables.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Paths to all generated deliverables.</returns>
    Task<DeliverableManifest> GenerateAllAsync(
        IncrementalWorkflowState state,
        DataFrame cleanedData,
        string outputDirectory,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Saves the cleaned DataFrame as a CSV file.
    /// </summary>
    /// <param name="data">The cleaned DataFrame.</param>
    /// <param name="outputPath">Path to save the CSV file.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task SaveCleanedDataAsync(
        DataFrame data,
        string outputPath,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates a processing report in markdown format.
    /// </summary>
    /// <param name="state">The completed workflow state.</param>
    /// <param name="outputPath">Path to save the report.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task GenerateReportAsync(
        IncrementalWorkflowState state,
        string outputPath,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Generates workflow metadata as JSON.
    /// </summary>
    /// <param name="state">The completed workflow state.</param>
    /// <param name="outputPath">Path to save the metadata.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task GenerateMetadataAsync(
        IncrementalWorkflowState state,
        string outputPath,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Manifest of all generated deliverables with file paths.
/// </summary>
public sealed class DeliverableManifest
{
    /// <summary>
    /// Path to the cleaned CSV file.
    /// </summary>
    public required string CleanedDataPath { get; init; }

    /// <summary>
    /// Path to the generated C# script.
    /// </summary>
    public string? ScriptPath { get; init; }

    /// <summary>
    /// Path to the processing report (markdown).
    /// </summary>
    public string? ReportPath { get; init; }

    /// <summary>
    /// Path to the workflow metadata (JSON).
    /// </summary>
    public string? MetadataPath { get; init; }

    /// <summary>
    /// When the deliverables were generated.
    /// </summary>
    public DateTime GeneratedAt { get; init; } = DateTime.UtcNow;
}
