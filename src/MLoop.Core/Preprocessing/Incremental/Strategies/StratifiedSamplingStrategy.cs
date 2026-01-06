using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.Contracts;
using MLoop.Core.Preprocessing.Incremental.Models;

namespace MLoop.Core.Preprocessing.Incremental.Strategies;

/// <summary>
/// Stratified sampling strategy that preserves class distribution.
/// </summary>
/// <remarks>
/// <para>Maintains the proportion of each class in the label column.</para>
/// <para>Ideal for classification tasks to ensure balanced representation.</para>
/// <para>Requires a valid label column to be specified in configuration.</para>
/// </remarks>
public sealed class StratifiedSamplingStrategy : ISamplingStrategy
{
    /// <inheritdoc/>
    public string Name => "Stratified";

    /// <inheritdoc/>
    public bool IsApplicable(DataFrame data, SamplingConfiguration config)
    {
        // Requires label column
        if (string.IsNullOrEmpty(config.LabelColumn))
            return false;

        // Label column must exist
        if (!data.Columns.Any(c => c.Name == config.LabelColumn))
            return false;

        return true;
    }

    /// <inheritdoc/>
    public DataFrame Sample(DataFrame data, double sampleRatio, SamplingConfiguration config, int randomSeed = 42)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(config);

        if (sampleRatio <= 0 || sampleRatio > 1)
            throw new ArgumentOutOfRangeException(nameof(sampleRatio), "Sample ratio must be between 0 and 1");

        if (string.IsNullOrEmpty(config.LabelColumn))
            throw new ArgumentException("Label column must be specified for stratified sampling", nameof(config));

        // Handle edge cases
        if (sampleRatio == 1.0)
            return data.Clone();

        if (data.Rows.Count == 0)
            return data.Clone();

        var labelColumn = data.Columns[config.LabelColumn];
        if (labelColumn == null)
            throw new ArgumentException($"Label column '{config.LabelColumn}' not found", nameof(config));

        // Group indices by label value
        var stratumIndices = GroupByLabel(labelColumn);

        // Calculate sample size per stratum
        int totalSampleSize = Math.Max(1, (int)(data.Rows.Count * sampleRatio));
        var sampledIndices = new List<int>();

        var random = new Random(randomSeed);

        foreach (var (labelValue, indices) in stratumIndices)
        {
            // Calculate proportional sample size for this stratum
            double stratumProportion = (double)indices.Count / data.Rows.Count;
            int stratumSampleSize = Math.Max(1, (int)(totalSampleSize * stratumProportion));

            // Ensure we don't sample more than available
            stratumSampleSize = Math.Min(stratumSampleSize, indices.Count);

            // Randomly sample from this stratum
            var stratumSample = SampleFromStratum(indices, stratumSampleSize, random);
            sampledIndices.AddRange(stratumSample);
        }

        // Sort indices for efficient DataFrame access
        sampledIndices.Sort();

        // Create sampled DataFrame
        return CreateSampleDataFrame(data, sampledIndices.ToArray());
    }

    /// <inheritdoc/>
    public ValidationResult Validate(DataFrame source, DataFrame sample, SamplingConfiguration config)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(sample);
        ArgumentNullException.ThrowIfNull(config);

        var details = new Dictionary<string, object>();

        if (string.IsNullOrEmpty(config.LabelColumn))
        {
            return ValidationResult.Failure("Label column not specified for stratified sampling validation", details);
        }

        var sourceLabelColumn = source.Columns[config.LabelColumn];
        var sampleLabelColumn = sample.Columns[config.LabelColumn];

        if (sourceLabelColumn == null || sampleLabelColumn == null)
        {
            return ValidationResult.Failure($"Label column '{config.LabelColumn}' not found", details);
        }

        // Calculate class distributions
        var sourceDistribution = CalculateDistribution(sourceLabelColumn);
        var sampleDistribution = CalculateDistribution(sampleLabelColumn);

        details["SourceDistribution"] = sourceDistribution;
        details["SampleDistribution"] = sampleDistribution;

        // Validate distribution preservation
        var maxDifference = 0.0;
        var differences = new Dictionary<string, double>();

        foreach (var (labelValue, sourceProportion) in sourceDistribution)
        {
            var sampleProportion = sampleDistribution.GetValueOrDefault(labelValue, 0.0);
            var difference = Math.Abs(sourceProportion - sampleProportion);
            differences[labelValue] = difference;
            maxDifference = Math.Max(maxDifference, difference);
        }

        details["MaxDifference"] = maxDifference;
        details["Differences"] = differences;
        details["Tolerance"] = config.DistributionTolerance;

        bool isValid = maxDifference <= config.DistributionTolerance;

        return isValid
            ? ValidationResult.Success($"Stratified sampling preserved distribution (max diff: {maxDifference:P2})", details)
            : ValidationResult.Failure($"Distribution not preserved (max diff: {maxDifference:P2} > tolerance {config.DistributionTolerance:P2})", details);
    }

    /// <summary>
    /// Groups row indices by label value.
    /// </summary>
    private static Dictionary<string, List<int>> GroupByLabel(DataFrameColumn labelColumn)
    {
        var groups = new Dictionary<string, List<int>>();

        for (int i = 0; i < labelColumn.Length; i++)
        {
            var value = labelColumn[i]?.ToString() ?? "NULL";

            if (!groups.TryGetValue(value, out var indices))
            {
                indices = new List<int>();
                groups[value] = indices;
            }

            indices.Add(i);
        }

        return groups;
    }

    /// <summary>
    /// Randomly samples indices from a stratum.
    /// </summary>
    private static List<int> SampleFromStratum(List<int> indices, int sampleSize, Random random)
    {
        // Use reservoir sampling for efficiency
        var sample = new List<int>(sampleSize);

        // Initialize with first sampleSize elements
        for (int i = 0; i < Math.Min(sampleSize, indices.Count); i++)
        {
            sample.Add(indices[i]);
        }

        // Reservoir sampling for remaining elements
        for (int i = sampleSize; i < indices.Count; i++)
        {
            int j = random.Next(0, i + 1);
            if (j < sampleSize)
            {
                sample[j] = indices[i];
            }
        }

        return sample;
    }

    /// <summary>
    /// Calculates class distribution as proportions.
    /// </summary>
    private static Dictionary<string, double> CalculateDistribution(DataFrameColumn labelColumn)
    {
        var counts = new Dictionary<string, int>();
        int total = 0;

        for (int i = 0; i < labelColumn.Length; i++)
        {
            var value = labelColumn[i]?.ToString() ?? "NULL";
            counts[value] = counts.GetValueOrDefault(value, 0) + 1;
            total++;
        }

        var distribution = new Dictionary<string, double>();
        foreach (var (value, count) in counts)
        {
            distribution[value] = (double)count / total;
        }

        return distribution;
    }

    /// <summary>
    /// Creates a new DataFrame from selected row indices.
    /// </summary>
    private static DataFrame CreateSampleDataFrame(DataFrame source, int[] indices)
    {
        var sampleColumns = new List<DataFrameColumn>();

        foreach (var column in source.Columns)
        {
            var sampleColumn = CreateSampleColumn(column, indices);
            sampleColumns.Add(sampleColumn);
        }

        return new DataFrame(sampleColumns);
    }

    /// <summary>
    /// Creates a sampled column from selected indices.
    /// </summary>
    private static DataFrameColumn CreateSampleColumn(DataFrameColumn sourceColumn, int[] indices)
    {
        DataFrameColumn sampleColumn = sourceColumn switch
        {
            PrimitiveDataFrameColumn<int> intCol => new PrimitiveDataFrameColumn<int>(intCol.Name, indices.Length),
            PrimitiveDataFrameColumn<long> longCol => new PrimitiveDataFrameColumn<long>(longCol.Name, indices.Length),
            PrimitiveDataFrameColumn<float> floatCol => new PrimitiveDataFrameColumn<float>(floatCol.Name, indices.Length),
            PrimitiveDataFrameColumn<double> doubleCol => new PrimitiveDataFrameColumn<double>(doubleCol.Name, indices.Length),
            PrimitiveDataFrameColumn<bool> boolCol => new PrimitiveDataFrameColumn<bool>(boolCol.Name, indices.Length),
            StringDataFrameColumn stringCol => new StringDataFrameColumn(stringCol.Name, indices.Length),
            _ => throw new NotSupportedException($"Column type {sourceColumn.GetType()} not supported")
        };

        for (int i = 0; i < indices.Length; i++)
        {
            sampleColumn[i] = sourceColumn[indices[i]];
        }

        return sampleColumn;
    }
}
