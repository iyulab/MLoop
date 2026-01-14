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
    /// Checks if datasets directory exists
    /// </summary>
    bool HasDatasetsDirectory(string projectRoot);

    /// <summary>
    /// Gets the datasets directory path
    /// </summary>
    string GetDatasetsPath(string projectRoot);
}
