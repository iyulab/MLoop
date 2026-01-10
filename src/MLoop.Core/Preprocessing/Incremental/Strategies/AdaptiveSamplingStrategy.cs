using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.Contracts;
using MLoop.Core.Preprocessing.Incremental.Models;

namespace MLoop.Core.Preprocessing.Incremental.Strategies;

/// <summary>
/// Adaptive sampling strategy that automatically selects the best approach.
/// </summary>
/// <remarks>
/// <para>Decision logic:</para>
/// <list type="bullet">
/// <item><description>Stratified: If label column exists and has reasonable cardinality</description></item>
/// <item><description>Random: Otherwise</description></item>
/// </list>
/// <para>Adapts to dataset characteristics for optimal sampling.</para>
/// </remarks>
public sealed class AdaptiveSamplingStrategy : ISamplingStrategy
{
    private readonly StratifiedSamplingStrategy _stratifiedStrategy;
    private readonly RandomSamplingStrategy _randomStrategy;

    /// <summary>
    /// Initializes a new instance of the <see cref="AdaptiveSamplingStrategy"/> class.
    /// </summary>
    public AdaptiveSamplingStrategy()
    {
        _stratifiedStrategy = new StratifiedSamplingStrategy();
        _randomStrategy = new RandomSamplingStrategy();
    }

    /// <inheritdoc/>
    public string Name => "Adaptive";

    /// <inheritdoc/>
    public bool IsApplicable(DataFrame data, SamplingConfiguration config)
    {
        // Adaptive is always applicable as it falls back to random
        return true;
    }

    /// <inheritdoc/>
    public DataFrame Sample(DataFrame data, double sampleRatio, SamplingConfiguration config, int randomSeed = 42)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(config);

        // Select best strategy based on data characteristics
        var selectedStrategy = SelectStrategy(data, config);

        return selectedStrategy.Sample(data, sampleRatio, config, randomSeed);
    }

    /// <inheritdoc/>
    public ValidationResult Validate(DataFrame source, DataFrame sample, SamplingConfiguration config)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(sample);
        ArgumentNullException.ThrowIfNull(config);

        // Validate using the strategy that would have been selected
        var selectedStrategy = SelectStrategy(source, config);

        var result = selectedStrategy.Validate(source, sample, config);

        // Add adaptive strategy information
        var details = new Dictionary<string, object>(result.Details)
        {
            ["SelectedStrategy"] = selectedStrategy.Name,
            ["AdaptiveReason"] = GetSelectionReason(source, config)
        };

        return new ValidationResult
        {
            IsValid = result.IsValid,
            Message = $"Adaptive ({selectedStrategy.Name}): {result.Message}",
            Details = details
        };
    }

    /// <summary>
    /// Selects the best sampling strategy based on data characteristics.
    /// </summary>
    private ISamplingStrategy SelectStrategy(DataFrame data, SamplingConfiguration config)
    {
        // Check if stratified sampling is applicable and beneficial
        if (_stratifiedStrategy.IsApplicable(data, config))
        {
            var labelColumn = data.Columns[config.LabelColumn!];

            // Calculate label cardinality
            var uniqueValues = GetUniqueValueCount(labelColumn);
            var totalRows = data.Rows.Count;
            var cardinalityRatio = (double)uniqueValues / totalRows;

            // Use stratified if:
            // 1. Reasonable number of classes (2-100)
            // 2. Not too high cardinality (< 50% unique)
            // 3. Each class has sufficient samples
            if (uniqueValues >= 2 && uniqueValues <= 100 && cardinalityRatio < 0.5)
            {
                // Check minimum samples per class
                var minSamplesPerClass = totalRows / uniqueValues;
                if (minSamplesPerClass >= 5)
                {
                    return _stratifiedStrategy;
                }
            }
        }

        // Fall back to random sampling
        return _randomStrategy;
    }

    /// <summary>
    /// Gets the reason for strategy selection (for logging/debugging).
    /// </summary>
    private string GetSelectionReason(DataFrame data, SamplingConfiguration config)
    {
        if (_stratifiedStrategy.IsApplicable(data, config))
        {
            var labelColumn = data.Columns[config.LabelColumn!];
            var uniqueValues = GetUniqueValueCount(labelColumn);
            var totalRows = data.Rows.Count;
            var cardinalityRatio = (double)uniqueValues / totalRows;

            if (uniqueValues < 2)
                return "Only one class - random sampling used";

            if (uniqueValues > 100)
                return $"Too many classes ({uniqueValues}) - random sampling used";

            if (cardinalityRatio >= 0.5)
                return $"High cardinality ({cardinalityRatio:P0}) - random sampling used";

            var minSamplesPerClass = totalRows / uniqueValues;
            if (minSamplesPerClass < 5)
                return $"Insufficient samples per class ({minSamplesPerClass}) - random sampling used";

            return $"Stratified sampling used ({uniqueValues} classes, {cardinalityRatio:P1} cardinality)";
        }

        return "No label column - random sampling used";
    }

    /// <summary>
    /// Counts unique values in a column.
    /// </summary>
    private static int GetUniqueValueCount(DataFrameColumn column)
    {
        var uniqueValues = new HashSet<string>();

        for (int i = 0; i < column.Length; i++)
        {
            var value = column[i]?.ToString() ?? "NULL";
            uniqueValues.Add(value);
        }

        return uniqueValues.Count;
    }
}
