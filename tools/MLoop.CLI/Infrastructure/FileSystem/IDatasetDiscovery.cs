namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Interface for discovering dataset files following MLOps conventions
/// </summary>
public interface IDatasetDiscovery
{
    /// <summary>
    /// Finds dataset files in the datasets/ directory
    /// </summary>
    /// <param name="projectRoot">Project root directory</param>
    /// <returns>Dataset paths if found, null otherwise</returns>
    DatasetPaths? FindDatasets(string projectRoot);

    /// <summary>
    /// Finds the one conventional file a <paramref name="role"/> reads under <c>datasets/</c> (<c>train.csv</c>, <c>test.csv</c>,
    /// <c>predict.csv</c>), whichever of the others exist — unlike <see cref="FindDatasets"/>, which is for training and answers
    /// nothing without <c>train.csv</c>. A project that only scores or predicts (its training data lives elsewhere) has no train.csv.
    /// </summary>
    string? FindFile(string projectRoot, DatasetRole role);

    /// <summary>
    /// Checks if datasets directory exists
    /// </summary>
    bool HasDatasetsDirectory(string projectRoot);

    /// <summary>
    /// Gets the datasets directory path
    /// </summary>
    string GetDatasetsPath(string projectRoot);
}
