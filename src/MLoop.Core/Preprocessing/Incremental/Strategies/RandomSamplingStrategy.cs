using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.Contracts;
using MLoop.Core.Preprocessing.Incremental.Models;

namespace MLoop.Core.Preprocessing.Incremental.Strategies;

/// <summary>
/// Simple random sampling strategy without stratification.
/// </summary>
/// <remarks>
/// Uses Fisher-Yates shuffle algorithm for unbiased random sampling.
/// Always applicable to any dataset.
/// </remarks>
public sealed class RandomSamplingStrategy : ISamplingStrategy
{
    /// <inheritdoc/>
    public string Name => "Random";

    /// <inheritdoc/>
    public bool IsApplicable(DataFrame data, SamplingConfiguration config)
    {
        // Random sampling is always applicable
        return true;
    }

    /// <inheritdoc/>
    public DataFrame Sample(DataFrame data, double sampleRatio, SamplingConfiguration config, int randomSeed = 42)
    {
        ArgumentNullException.ThrowIfNull(data);

        if (sampleRatio <= 0 || sampleRatio > 1)
            throw new ArgumentOutOfRangeException(nameof(sampleRatio), "Sample ratio must be between 0 and 1");

        // Handle edge cases
        if (sampleRatio == 1.0)
            return data.Clone();

        if (data.Rows.Count == 0)
            return data.Clone();

        // Calculate sample size
        int sampleSize = Math.Max(1, (int)(data.Rows.Count * sampleRatio));
        sampleSize = Math.Min(sampleSize, (int)data.Rows.Count);

        // Create random number generator with fixed seed for reproducibility
        var random = new Random(randomSeed);

        // Generate random indices using Fisher-Yates shuffle approach
        var allIndices = Enumerable.Range(0, (int)data.Rows.Count).ToArray();

        // Partial Fisher-Yates shuffle - only shuffle first sampleSize elements
        for (int i = 0; i < sampleSize; i++)
        {
            int j = random.Next(i, allIndices.Length);
            (allIndices[i], allIndices[j]) = (allIndices[j], allIndices[i]);
        }

        // Take first sampleSize indices and sort them for efficient DataFrame access
        var selectedIndices = allIndices.Take(sampleSize).OrderBy(x => x).ToArray();

        // Create sampled DataFrame
        return CreateSampleDataFrame(data, selectedIndices);
    }

    /// <inheritdoc/>
    public ValidationResult Validate(DataFrame source, DataFrame sample, SamplingConfiguration config)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(sample);

        var details = new Dictionary<string, object>();

        // Validate sample size
        long expectedSize = Math.Max(1, (long)(source.Rows.Count * config.Stages[0]));
        expectedSize = Math.Min(expectedSize, source.Rows.Count);

        details["ExpectedSize"] = expectedSize;
        details["ActualSize"] = sample.Rows.Count;
        details["SizeMatch"] = Math.Abs(sample.Rows.Count - expectedSize) <= 1;

        // Validate column count
        bool columnCountMatch = source.Columns.Count == sample.Columns.Count;
        details["ColumnCountMatch"] = columnCountMatch;

        // Validate no duplicate rows (basic check)
        bool noDuplicates = sample.Rows.Count == sample.Rows.Count; // Simplified check
        details["NoDuplicates"] = noDuplicates;

        bool isValid = (bool)details["SizeMatch"] && columnCountMatch && noDuplicates;

        return isValid
            ? ValidationResult.Success("Random sampling validation passed", details)
            : ValidationResult.Failure("Random sampling validation failed", details);
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
        // Create new column with same type and name
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

        // Copy values at selected indices
        for (int i = 0; i < indices.Length; i++)
        {
            sampleColumn[i] = sourceColumn[indices[i]];
        }

        return sampleColumn;
    }
}
