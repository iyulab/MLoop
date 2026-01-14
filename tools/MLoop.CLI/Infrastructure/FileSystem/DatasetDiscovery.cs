namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// MLOps convention-based dataset discovery
/// Looks for datasets/ directory with train.csv, validation.csv, test.csv
/// </summary>
public class DatasetDiscovery : IDatasetDiscovery
{
    private const string DatasetsDirectoryName = "datasets";
    private const string TrainFileName = "train.csv";
    private const string ValidationFileName = "validation.csv";
    private const string TestFileName = "test.csv";
    private const string PredictFileName = "predict.csv";

    private readonly IFileSystemManager _fileSystem;

    public DatasetDiscovery(IFileSystemManager fileSystem)
    {
        _fileSystem = fileSystem ?? throw new ArgumentNullException(nameof(fileSystem));
    }

    public DatasetPaths? FindDatasets(string projectRoot)
    {
        var datasetsPath = GetDatasetsPath(projectRoot);

        if (!_fileSystem.DirectoryExists(datasetsPath))
        {
            return null;
        }

        var trainPath = _fileSystem.CombinePath(datasetsPath, TrainFileName);
        if (!_fileSystem.FileExists(trainPath))
        {
            return null; // train.csv is mandatory
        }

        return new DatasetPaths
        {
            TrainPath = trainPath,
            ValidationPath = TryFindFile(datasetsPath, ValidationFileName),
            TestPath = TryFindFile(datasetsPath, TestFileName),
            PredictPath = TryFindFile(datasetsPath, PredictFileName)
        };
    }

    public bool HasDatasetsDirectory(string projectRoot)
    {
        return _fileSystem.DirectoryExists(GetDatasetsPath(projectRoot));
    }

    public string GetDatasetsPath(string projectRoot)
    {
        return _fileSystem.CombinePath(projectRoot, DatasetsDirectoryName);
    }

    private string? TryFindFile(string directory, string fileName)
    {
        var filePath = _fileSystem.CombinePath(directory, fileName);
        return _fileSystem.FileExists(filePath) ? filePath : null;
    }
}
