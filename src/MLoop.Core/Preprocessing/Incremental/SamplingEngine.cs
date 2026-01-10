using Microsoft.Data.Analysis;
using Microsoft.Extensions.Logging;
using MLoop.Core.Preprocessing.Incremental.Contracts;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.Strategies;

namespace MLoop.Core.Preprocessing.Incremental;

/// <summary>
/// Main sampling engine for progressive dataset sampling.
/// </summary>
/// <remarks>
/// <para>Supports multiple sampling strategies:</para>
/// <list type="bullet">
/// <item><description>Auto: Automatically selects best strategy</description></item>
/// <item><description>Stratified: Preserves class distribution</description></item>
/// <item><description>Random: Simple random sampling</description></item>
/// <item><description>Adaptive: Dynamic strategy selection</description></item>
/// </list>
/// <para>Features:</para>
/// <list type="bullet">
/// <item><description>Deterministic with fixed random seed</description></item>
/// <item><description>Progress reporting for long operations</description></item>
/// <item><description>Validation of sampling quality</description></item>
/// <item><description>Memory efficient for large datasets</description></item>
/// </list>
/// </remarks>
public sealed class SamplingEngine : ISamplingEngine
{
    private readonly ILogger<SamplingEngine>? _logger;
    private ISamplingStrategy _strategy;

    /// <summary>
    /// Initializes a new instance of the <see cref="SamplingEngine"/> class.
    /// </summary>
    /// <param name="strategy">The sampling strategy to use. If null, adaptive strategy is used.</param>
    /// <param name="logger">Optional logger for diagnostics.</param>
    public SamplingEngine(ISamplingStrategy? strategy = null, ILogger<SamplingEngine>? logger = null)
    {
        _strategy = strategy ?? new AdaptiveSamplingStrategy();
        _logger = logger;
    }

    /// <inheritdoc/>
    public ISamplingStrategy Strategy => _strategy;

    /// <inheritdoc/>
    public async Task<DataFrame> SampleAsync(
        DataFrame data,
        double sampleRatio,
        SamplingConfiguration? config = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(data);

        if (sampleRatio <= 0 || sampleRatio > 1)
            throw new ArgumentOutOfRangeException(nameof(sampleRatio), "Sample ratio must be between 0 and 1");

        config ??= new SamplingConfiguration();

        _logger?.LogInformation(
            "Starting sampling: {Rows} rows, ratio {Ratio:P2}, strategy {Strategy}",
            data.Rows.Count, sampleRatio, _strategy.Name);

        // Select strategy based on configuration
        SelectStrategy(data, config);

        progress?.Report(0.0);

        // Perform sampling (async wrapper for sync operation)
        var sample = await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            return _strategy.Sample(data, sampleRatio, config, config.RandomSeed);
        }, cancellationToken);

        progress?.Report(0.5);

        // Validate sample
        var validation = _strategy.Validate(data, sample, config);

        if (!validation.IsValid)
        {
            _logger?.LogWarning(
                "Sampling validation failed: {Message}. Details: {@Details}",
                validation.Message, validation.Details);
        }
        else
        {
            _logger?.LogInformation(
                "Sampling completed: {SampleRows}/{TotalRows} rows ({Ratio:P2}), validation: {Message}",
                sample.Rows.Count, data.Rows.Count, sampleRatio, validation.Message);
        }

        progress?.Report(1.0);

        return sample;
    }

    /// <inheritdoc/>
    public bool ValidateSample(DataFrame source, DataFrame sample, double tolerance = 0.02)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(sample);

        var config = new SamplingConfiguration
        {
            DistributionTolerance = tolerance
        };

        var validation = _strategy.Validate(source, sample, config);

        return validation.IsValid;
    }

    /// <summary>
    /// Selects and sets the appropriate sampling strategy based on configuration.
    /// </summary>
    private void SelectStrategy(DataFrame data, SamplingConfiguration config)
    {
        _strategy = config.Strategy switch
        {
            SamplingStrategyType.Auto => SelectAutoStrategy(data, config),
            SamplingStrategyType.Stratified => new StratifiedSamplingStrategy(),
            SamplingStrategyType.Random => new RandomSamplingStrategy(),
            SamplingStrategyType.Adaptive => new AdaptiveSamplingStrategy(),
            _ => throw new ArgumentException($"Unknown sampling strategy: {config.Strategy}", nameof(config))
        };

        _logger?.LogDebug("Selected sampling strategy: {Strategy}", _strategy.Name);
    }

    /// <summary>
    /// Auto-selects the best strategy based on data and configuration.
    /// </summary>
    private static ISamplingStrategy SelectAutoStrategy(DataFrame data, SamplingConfiguration config)
    {
        // If label column is specified and valid, try stratified
        if (!string.IsNullOrEmpty(config.LabelColumn))
        {
            var stratified = new StratifiedSamplingStrategy();
            if (stratified.IsApplicable(data, config))
            {
                return stratified;
            }
        }

        // Fall back to random
        return new RandomSamplingStrategy();
    }
}
