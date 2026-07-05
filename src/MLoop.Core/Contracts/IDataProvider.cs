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
    IDataView LoadData(string filePath, string? labelColumn = null, string? taskType = null,
        IEnumerable<string>? preserveColumns = null);

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

    /// <summary>
    /// After <see cref="LoadData"/>, maps each merged vector column the loader produced
    /// (e.g. InferColumns' ranged "Features" vector) to the original source column names it
    /// spans, in slot order. The loaded IDataView carries no slot-name annotations for these
    /// columns, so this is the only way schema capture can recover the named columns that
    /// row-based prediction needs to reconstruct the vector (upstream-008). Null when the
    /// provider merged nothing or doesn't track merges.
    /// </summary>
    IReadOnlyDictionary<string, string[]>? GetMergedColumnGroups() => null;
}
