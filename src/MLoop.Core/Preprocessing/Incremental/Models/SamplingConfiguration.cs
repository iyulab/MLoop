namespace MLoop.Core.Preprocessing.Incremental.Models;

/// <summary>
/// Configuration for progressive sampling behavior.
/// </summary>
public sealed class SamplingConfiguration
{
    /// <summary>
    /// Gets or sets the label column name for stratified sampling.
    /// </summary>
    /// <remarks>
    /// If specified, stratified sampling will be used to preserve class distribution.
    /// If null, random sampling will be used.
    /// </remarks>
    public string? LabelColumn { get; set; }

    /// <summary>
    /// Gets or sets the sampling stages as ratios (0.0 to 1.0).
    /// </summary>
    /// <remarks>
    /// <para>Default stages: [0.001, 0.005, 0.015, 0.025, 1.0]</para>
    /// <para>Represents: 0.1%, 0.5%, 1.5%, 2.5%, 100%</para>
    /// <para>Custom stages can be specified for different progressions.</para>
    /// </remarks>
    public double[] Stages { get; set; } = [0.001, 0.005, 0.015, 0.025, 1.0];

    /// <summary>
    /// Gets or sets the sampling strategy to use.
    /// </summary>
    /// <remarks>
    /// <para>Options:</para>
    /// <list type="bullet">
    /// <item><description><b>Auto</b>: Automatically selects stratified if label column exists, otherwise random</description></item>
    /// <item><description><b>Stratified</b>: Preserves class distribution (requires label column)</description></item>
    /// <item><description><b>Random</b>: Simple random sampling</description></item>
    /// <item><description><b>Adaptive</b>: Adjusts strategy based on dataset characteristics</description></item>
    /// </list>
    /// </remarks>
    public SamplingStrategyType Strategy { get; set; } = SamplingStrategyType.Auto;

    /// <summary>
    /// Gets or sets the maximum number of stages to execute.
    /// </summary>
    /// <remarks>
    /// Useful for limiting execution time or when convergence is expected early.
    /// Default is 5 stages.
    /// </remarks>
    public int MaxStages { get; set; } = 5;

    /// <summary>
    /// Gets or sets the confidence threshold for rule discovery (used in later phases).
    /// </summary>
    /// <remarks>
    /// Rules with confidence below this threshold will require human confirmation.
    /// Default is 0.8 (80% confidence).
    /// </remarks>
    public double ConfidenceThreshold { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the tolerance for distribution preservation in stratified sampling.
    /// </summary>
    /// <remarks>
    /// Class distribution differences within this tolerance are acceptable.
    /// Default is 0.02 (2% tolerance).
    /// </remarks>
    public double DistributionTolerance { get; set; } = 0.02;

    /// <summary>
    /// Gets or sets whether to enable early stopping when convergence is detected.
    /// </summary>
    /// <remarks>
    /// If true, stops sampling after convergence is detected, saving time.
    /// Default is true.
    /// </remarks>
    public bool EnableEarlyStopping { get; set; } = true;

    /// <summary>
    /// Gets or sets the variance threshold for convergence detection.
    /// </summary>
    /// <remarks>
    /// When variance of key statistics falls below this threshold, convergence is detected.
    /// Default is 0.01 (1% variance).
    /// </remarks>
    public double ConvergenceThreshold { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    /// <remarks>
    /// Fixed seed ensures deterministic sampling results.
    /// Default is 42.
    /// </remarks>
    public int RandomSeed { get; set; } = 42;
}

/// <summary>
/// Sampling strategy types.
/// </summary>
public enum SamplingStrategyType
{
    /// <summary>
    /// Automatically select best strategy based on data characteristics.
    /// </summary>
    Auto,

    /// <summary>
    /// Stratified sampling - preserves class distribution.
    /// </summary>
    Stratified,

    /// <summary>
    /// Simple random sampling without stratification.
    /// </summary>
    Random,

    /// <summary>
    /// Adaptive sampling - adjusts strategy dynamically.
    /// </summary>
    Adaptive
}
