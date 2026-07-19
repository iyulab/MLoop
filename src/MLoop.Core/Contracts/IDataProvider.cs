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
    /// <param name="featureExclusions">
    /// Columns featurization must drop, decided once for the whole run (see
    /// <see cref="Data.CsvDataLoader.DetermineExcludedColumns"/>). When supplied the loader applies
    /// this set verbatim instead of re-deriving it from <paramref name="filePath"/> — the two
    /// partitions of one split must agree on the feature width, and a data-dependent rule evaluated
    /// per partition does not. Null means "no decision has been made yet; decide from this file".
    /// Loaders whose input is not tabular (image folders, COCO/YOLO annotations) have no such
    /// columns and ignore it, as they do <see cref="GetMergedColumnGroups"/>.
    /// </param>
    IDataView LoadData(string filePath, string? labelColumn = null, string? taskType = null,
        IEnumerable<string>? preserveColumns = null, IReadOnlyCollection<string>? featureExclusions = null);

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
