namespace MLoop.Core.Preprocessing.Incremental.Models;

/// <summary>
/// Configuration for the incremental preprocessing workflow.
/// </summary>
public sealed class IncrementalWorkflowConfig
{
    // ===== Sampling Ratios =====

    /// <summary>
    /// Stage 1: Initial exploration sample ratio (default 0.1%).
    /// </summary>
    public double Stage1Ratio { get; init; } = 0.001;

    /// <summary>
    /// Stage 2: Pattern expansion sample ratio (default 0.5%).
    /// </summary>
    public double Stage2Ratio { get; init; } = 0.005;

    /// <summary>
    /// Stage 3: HITL decision sample ratio (default 1.5%).
    /// </summary>
    public double Stage3Ratio { get; init; } = 0.015;

    /// <summary>
    /// Stage 4: Confidence checkpoint sample ratio (default 2.5%).
    /// </summary>
    public double Stage4Ratio { get; init; } = 0.025;

    // ===== Quality Thresholds =====

    /// <summary>
    /// Minimum confidence threshold for rule approval (default 0.98).
    /// Rules below this confidence require additional validation.
    /// </summary>
    public double MinConfidenceThreshold { get; init; } = 0.98;

    /// <summary>
    /// Maximum acceptable error rate during bulk processing (default 0.01).
    /// Processing stops if error rate exceeds this threshold.
    /// </summary>
    public double MaxErrorRate { get; init; } = 0.01;

    // ===== HITL Options =====

    /// <summary>
    /// If true, skips HITL prompts and uses recommended defaults.
    /// Useful for automated pipelines.
    /// </summary>
    public bool SkipHITL { get; init; } = false;

    /// <summary>
    /// If true, automatically approves rules at confidence checkpoint.
    /// Requires MinConfidenceThreshold to be met.
    /// </summary>
    public bool EnableAutoApproval { get; init; } = false;

    // ===== Output Configuration =====

    /// <summary>
    /// Directory where cleaned data and deliverables are saved.
    /// </summary>
    public string OutputDirectory { get; init; } = "./cleaned";

    /// <summary>
    /// If true, generates reusable C# preprocessing scripts.
    /// </summary>
    public bool GenerateScripts { get; init; } = true;

    /// <summary>
    /// If true, generates detailed processing report (markdown).
    /// </summary>
    public bool GenerateReport { get; init; } = true;

    // ===== Checkpoint Management =====

    /// <summary>
    /// If true, saves workflow state after each stage for resume capability.
    /// </summary>
    public bool EnableCheckpoints { get; init; } = true;

    /// <summary>
    /// Directory where workflow checkpoints are saved.
    /// </summary>
    public string CheckpointDirectory { get; init; } = "./checkpoints";

    // ===== Processing Options =====

    /// <summary>
    /// Chunk size for bulk processing (default 10,000 records).
    /// Larger chunks = faster processing, more memory usage.
    /// </summary>
    public int BulkProcessingChunkSize { get; init; } = 10000;

    /// <summary>
    /// If true, continues processing even if individual rules fail.
    /// Failed rules are logged and skipped.
    /// </summary>
    public bool ContinueOnRuleFailure { get; init; } = true;
}
