using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLoop.Core.Contracts;
using MLoop.Core.Models;

namespace MLoop.Core.Data;

/// <summary>
/// CSV data loader implementation using ML.NET
/// </summary>
public class CsvDataLoader : IDataProvider
{
    private readonly MLContext _mlContext;

    public CsvDataLoader(MLContext mlContext)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
    }

    public IDataView LoadData(string filePath, string? labelColumn = null)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Data file not found: {filePath}");
        }

        // Infer columns from the file
        var columnInference = _mlContext.Auto().InferColumns(
            filePath,
            labelColumnName: labelColumn,
            separatorChar: ',');

        // Create text loader with inferred schema
        var loader = _mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
        var dataView = loader.Load(filePath);

        // Validate label column if specified
        if (!string.IsNullOrEmpty(labelColumn))
        {
            if (!ValidateLabelColumn(dataView, labelColumn))
            {
                throw new InvalidOperationException(
                    $"Label column '{labelColumn}' not found in the data. " +
                    $"Available columns: {string.Join(", ", GetColumnNames(dataView))}");
            }
        }

        return dataView;
    }

    public bool ValidateLabelColumn(IDataView data, string labelColumn)
    {
        var schema = data.Schema;
        return schema.Any(col => col.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase));
    }

    public DataSchema GetSchema(IDataView data)
    {
        var schema = data.Schema;
        var columns = new List<ColumnInfo>();

        foreach (var column in schema)
        {
            columns.Add(new ColumnInfo
            {
                Name = column.Name,
                Type = DataTypeHelper.GetFriendlyTypeName(column.Type),
                IsLabel = false
            });
        }

        // Get approximate row count
        var rowCount = GetRowCount(data);

        return new DataSchema
        {
            Columns = columns,
            RowCount = rowCount
        };
    }

    public (IDataView trainSet, IDataView testSet) SplitData(
        IDataView data,
        double testFraction = 0.2)
    {
        // If testFraction is 0, use all data for both train and test (AutoML will use cross-validation)
        if (testFraction <= 0)
        {
            return (data, data);
        }

        if (testFraction >= 1)
        {
            throw new ArgumentException(
                "Test fraction must be less than 1",
                nameof(testFraction));
        }

        var split = _mlContext.Data.TrainTestSplit(
            data,
            testFraction: testFraction,
            seed: 42); // Fixed seed for reproducibility

        return (split.TrainSet, split.TestSet);
    }

    private IEnumerable<string> GetColumnNames(IDataView data)
    {
        return data.Schema.Select(col => col.Name);
    }

    private int GetRowCount(IDataView data)
    {
        // Try to get row count efficiently
        if (data is IDataView view)
        {
            long? count = view.GetRowCount();
            if (count.HasValue)
            {
                return (int)count.Value;
            }
        }

        // Fallback: count manually (expensive)
        int rowCount = 0;
        using (var cursor = data.GetRowCursor(data.Schema))
        {
            while (cursor.MoveNext())
            {
                rowCount++;
            }
        }

        return rowCount;
    }
}
