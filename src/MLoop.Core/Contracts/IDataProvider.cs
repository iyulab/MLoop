using Microsoft.ML;
using MLoop.Core.Models;

namespace MLoop.Core.Contracts;

/// <summary>
/// Loads and validates data for ML.NET
/// </summary>
public interface IDataProvider
{
    /// <summary>
    /// Loads data from a file path
    /// </summary>
    IDataView LoadData(string filePath, string? labelColumn = null);

    /// <summary>
    /// Validates that the label column exists in the data
    /// </summary>
    bool ValidateLabelColumn(IDataView data, string labelColumn);

    /// <summary>
    /// Gets the schema information from the data
    /// </summary>
    DataSchema GetSchema(IDataView data);

    /// <summary>
    /// Splits data into training and test sets
    /// </summary>
    (IDataView trainSet, IDataView testSet) SplitData(
        IDataView data,
        double testFraction = 0.2);
}
