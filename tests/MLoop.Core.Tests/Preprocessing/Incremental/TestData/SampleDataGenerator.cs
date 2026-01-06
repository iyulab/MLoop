using Microsoft.Data.Analysis;

namespace MLoop.Core.Tests.Preprocessing.Incremental.TestData;

/// <summary>
/// Utility class for generating test DataFrames with various characteristics.
/// </summary>
public static class SampleDataGenerator
{
    /// <summary>
    /// Generates a DataFrame with numeric and categorical columns.
    /// </summary>
    /// <param name="rowCount">Number of rows to generate.</param>
    /// <param name="labelColumn">Name of the label column for classification.</param>
    /// <param name="numClasses">Number of unique classes in label column.</param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    public static DataFrame GenerateMixedData(
        int rowCount,
        string labelColumn = "Label",
        int numClasses = 3,
        int randomSeed = 42)
    {
        var random = new Random(randomSeed);
        var dataFrame = new DataFrame();

        // Numeric columns
        var numericColumn1 = new PrimitiveDataFrameColumn<double>("Feature1",
            Enumerable.Range(0, rowCount).Select(_ => random.NextDouble() * 100).ToArray());

        var numericColumn2 = new PrimitiveDataFrameColumn<double>("Feature2",
            Enumerable.Range(0, rowCount).Select(_ => random.NextDouble() * 50 + 25).ToArray());

        var numericColumn3 = new PrimitiveDataFrameColumn<int>("Feature3",
            Enumerable.Range(0, rowCount).Select(_ => random.Next(0, 1000)).ToArray());

        // Categorical column (label)
        var labels = new StringDataFrameColumn(labelColumn,
            Enumerable.Range(0, rowCount).Select(_ => $"Class{random.Next(0, numClasses)}").ToArray());

        // Categorical column (category)
        var categories = new StringDataFrameColumn("Category",
            Enumerable.Range(0, rowCount).Select(_ =>
                new[] { "A", "B", "C", "D", "E" }[random.Next(0, 5)]).ToArray());

        // Additional test columns for HITL scenarios
        var age = new PrimitiveDataFrameColumn<int>("Age",
            Enumerable.Range(0, rowCount).Select(_ => random.Next(18, 80)).ToArray());

        var salary = new PrimitiveDataFrameColumn<double>("Salary",
            Enumerable.Range(0, rowCount).Select(_ => random.NextDouble() * 100000 + 30000).ToArray());

        dataFrame.Columns.Add(numericColumn1);
        dataFrame.Columns.Add(numericColumn2);
        dataFrame.Columns.Add(numericColumn3);
        dataFrame.Columns.Add(age);
        dataFrame.Columns.Add(salary);
        dataFrame.Columns.Add(categories);
        dataFrame.Columns.Add(labels);

        return dataFrame;
    }

    /// <summary>
    /// Generates a DataFrame with missing values.
    /// </summary>
    /// <param name="rowCount">Number of rows to generate.</param>
    /// <param name="missingPercentage">Percentage of missing values (0.0 to 1.0).</param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    public static DataFrame GenerateDataWithMissing(
        int rowCount,
        double missingPercentage = 0.2,
        int randomSeed = 42)
    {
        var random = new Random(randomSeed);
        var dataFrame = new DataFrame();

        var numericColumn = new PrimitiveDataFrameColumn<double>("Feature1");
        for (int i = 0; i < rowCount; i++)
        {
            if (random.NextDouble() < missingPercentage)
                numericColumn.Append(null);
            else
                numericColumn.Append(random.NextDouble() * 100);
        }

        var categoricalColumn = new StringDataFrameColumn("Category");
        for (int i = 0; i < rowCount; i++)
        {
            if (random.NextDouble() < missingPercentage)
                categoricalColumn.Append(null);
            else
                categoricalColumn.Append(new[] { "A", "B", "C" }[random.Next(0, 3)]);
        }

        dataFrame.Columns.Add(numericColumn);
        dataFrame.Columns.Add(categoricalColumn);

        return dataFrame;
    }

    /// <summary>
    /// Generates a DataFrame with outliers in numeric columns.
    /// </summary>
    /// <param name="rowCount">Number of rows to generate.</param>
    /// <param name="outlierPercentage">Percentage of outliers (0.0 to 1.0).</param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    public static DataFrame GenerateDataWithOutliers(
        int rowCount,
        double outlierPercentage = 0.05,
        int randomSeed = 42)
    {
        var random = new Random(randomSeed);
        var dataFrame = new DataFrame();

        var numericColumn = new PrimitiveDataFrameColumn<double>("Feature1");
        for (int i = 0; i < rowCount; i++)
        {
            if (random.NextDouble() < outlierPercentage)
            {
                // Generate outlier (very large value)
                numericColumn.Append(random.NextDouble() * 10000 + 1000);
            }
            else
            {
                // Normal value (0-100)
                numericColumn.Append(random.NextDouble() * 100);
            }
        }

        dataFrame.Columns.Add(numericColumn);
        return dataFrame;
    }

    /// <summary>
    /// Generates a DataFrame with high cardinality categorical column.
    /// </summary>
    /// <param name="rowCount">Number of rows to generate.</param>
    /// <param name="uniqueValues">Number of unique values in categorical column.</param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    public static DataFrame GenerateHighCardinalityData(
        int rowCount,
        int uniqueValues = 500,
        int randomSeed = 42)
    {
        var random = new Random(randomSeed);
        var dataFrame = new DataFrame();

        var categoricalColumn = new StringDataFrameColumn("HighCardinalityColumn",
            Enumerable.Range(0, rowCount).Select(_ => $"Value{random.Next(0, uniqueValues)}").ToArray());

        dataFrame.Columns.Add(categoricalColumn);
        return dataFrame;
    }

    /// <summary>
    /// Generates a DataFrame with balanced class distribution.
    /// </summary>
    /// <param name="rowCount">Number of rows to generate.</param>
    /// <param name="numClasses">Number of classes.</param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    public static DataFrame GenerateBalancedData(
        int rowCount,
        int numClasses = 3,
        int randomSeed = 42)
    {
        var random = new Random(randomSeed);
        var dataFrame = new DataFrame();

        // Ensure balanced distribution
        var labels = new List<string>();
        int samplesPerClass = rowCount / numClasses;
        int remainder = rowCount % numClasses;

        for (int i = 0; i < numClasses; i++)
        {
            int count = samplesPerClass + (i < remainder ? 1 : 0);
            labels.AddRange(Enumerable.Repeat($"Class{i}", count));
        }

        // Shuffle labels
        for (int i = labels.Count - 1; i > 0; i--)
        {
            int j = random.Next(0, i + 1);
            (labels[i], labels[j]) = (labels[j], labels[i]);
        }

        var labelColumn = new StringDataFrameColumn("Label", labels);

        var numericColumn = new PrimitiveDataFrameColumn<double>("Feature1",
            Enumerable.Range(0, rowCount).Select(_ => random.NextDouble() * 100).ToArray());

        dataFrame.Columns.Add(numericColumn);
        dataFrame.Columns.Add(labelColumn);

        return dataFrame;
    }

    /// <summary>
    /// Generates a DataFrame with imbalanced class distribution.
    /// </summary>
    /// <param name="rowCount">Number of rows to generate.</param>
    /// <param name="numClasses">Number of classes.</param>
    /// <param name="majorityClassRatio">Ratio of majority class (0.0 to 1.0).</param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    public static DataFrame GenerateImbalancedData(
        int rowCount,
        int numClasses = 3,
        double majorityClassRatio = 0.7,
        int randomSeed = 42)
    {
        var random = new Random(randomSeed);
        var dataFrame = new DataFrame();

        var labels = new List<string>();
        int majorityCount = (int)(rowCount * majorityClassRatio);
        int minorityCount = (rowCount - majorityCount) / (numClasses - 1);
        int remainder = (rowCount - majorityCount) % (numClasses - 1);

        // Majority class
        labels.AddRange(Enumerable.Repeat("Class0", majorityCount));

        // Minority classes
        for (int i = 1; i < numClasses; i++)
        {
            int count = minorityCount + (i - 1 < remainder ? 1 : 0);
            labels.AddRange(Enumerable.Repeat($"Class{i}", count));
        }

        // Shuffle labels
        for (int i = labels.Count - 1; i > 0; i--)
        {
            int j = random.Next(0, i + 1);
            (labels[i], labels[j]) = (labels[j], labels[i]);
        }

        var labelColumn = new StringDataFrameColumn("Label", labels);

        var numericColumn = new PrimitiveDataFrameColumn<double>("Feature1",
            Enumerable.Range(0, rowCount).Select(_ => random.NextDouble() * 100).ToArray());

        dataFrame.Columns.Add(numericColumn);
        dataFrame.Columns.Add(labelColumn);

        return dataFrame;
    }

    /// <summary>
    /// Generates an empty DataFrame.
    /// </summary>
    public static DataFrame GenerateEmptyData()
    {
        var dataFrame = new DataFrame();
        dataFrame.Columns.Add(new PrimitiveDataFrameColumn<double>("Feature1", Array.Empty<double>()));
        dataFrame.Columns.Add(new StringDataFrameColumn("Label", Array.Empty<string>()));
        return dataFrame;
    }

    /// <summary>
    /// Generates a DataFrame with a single row.
    /// </summary>
    public static DataFrame GenerateSingleRowData()
    {
        var dataFrame = new DataFrame();
        dataFrame.Columns.Add(new PrimitiveDataFrameColumn<double>("Feature1", new[] { 42.0 }));
        dataFrame.Columns.Add(new StringDataFrameColumn("Label", new[] { "Class0" }));
        return dataFrame;
    }

    /// <summary>
    /// Generates a DataFrame with all same values.
    /// </summary>
    /// <param name="rowCount">Number of rows to generate.</param>
    public static DataFrame GenerateSameValueData(int rowCount = 100)
    {
        var dataFrame = new DataFrame();
        dataFrame.Columns.Add(new PrimitiveDataFrameColumn<double>("Feature1",
            Enumerable.Repeat(42.0, rowCount).ToArray()));
        dataFrame.Columns.Add(new StringDataFrameColumn("Label",
            Enumerable.Repeat("Class0", rowCount).ToArray()));
        return dataFrame;
    }

    /// <summary>
    /// Generates large-scale test data for performance testing.
    /// </summary>
    /// <param name="rowCount">Number of rows (e.g., 1_000_000).</param>
    /// <param name="columnCount">Number of numeric columns.</param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    public static DataFrame GenerateLargeScaleData(
        int rowCount,
        int columnCount = 10,
        int randomSeed = 42)
    {
        var random = new Random(randomSeed);
        var dataFrame = new DataFrame();

        // Add numeric columns
        for (int col = 0; col < columnCount; col++)
        {
            var column = new PrimitiveDataFrameColumn<double>($"Feature{col}",
                Enumerable.Range(0, rowCount).Select(_ => random.NextDouble() * 100).ToArray());
            dataFrame.Columns.Add(column);
        }

        // Add label column
        var labels = new StringDataFrameColumn("Label",
            Enumerable.Range(0, rowCount).Select(_ => $"Class{random.Next(0, 3)}").ToArray());
        dataFrame.Columns.Add(labels);

        return dataFrame;
    }
}
