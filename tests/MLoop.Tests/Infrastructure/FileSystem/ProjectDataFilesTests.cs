using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.Infrastructure.FileSystem;

/// <summary>
/// The default data file of a role: the file <c>mloop.yaml</c> declares when it exists, else the conventional one - and the
/// conventional one is found whichever of the others the project has.
/// </summary>
public sealed class ProjectDataFilesTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "mloop-data-files-" + Guid.NewGuid().ToString("N"));
    private readonly DatasetDiscovery _discovery = new(new FileSystemManager());

    public ProjectDataFilesTests() => Directory.CreateDirectory(Path.Combine(_root, "datasets"));

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private string Write(string relative)
    {
        var path = Path.Combine(_root, relative);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, "a,b\n1,2\n");
        return path;
    }

    private static MLoopConfig Declaring(string? train = null, string? test = null, string? predict = null)
        => new() { Data = new DataSettings { Train = train, Test = test, Predict = predict } };

    [Theory]
    [InlineData(DatasetRole.Train)]
    [InlineData(DatasetRole.Test)]
    [InlineData(DatasetRole.Predict)]
    public void ADeclaredFileThatExists_IsTheDefault_OverTheConventionalOne(DatasetRole role)
    {
        Write("datasets/train.csv"); Write("datasets/test.csv"); Write("datasets/predict.csv");
        var declared = Write("elsewhere/mine.csv");
        var config = Declaring(
            train: role == DatasetRole.Train ? "elsewhere/mine.csv" : null,
            test: role == DatasetRole.Test ? "elsewhere/mine.csv" : null,
            predict: role == DatasetRole.Predict ? "elsewhere/mine.csv" : null);

        var found = ProjectDataFiles.Resolve(role, config, _root, _discovery);

        Assert.Equal(declared, found.Path);
        Assert.Null(found.DeclaredMissing);
    }

    [Fact]
    public void ADeclaredFileThatDoesNotExist_FallsBackToTheConventionalOne()
    {
        var conventional = Write("datasets/test.csv");

        var found = ProjectDataFiles.Resolve(DatasetRole.Test, Declaring(test: "gone.csv"), _root, _discovery);

        Assert.Equal(conventional, found.Path);
        Assert.Null(found.DeclaredMissing);
    }

    [Fact]
    public void WithNeitherFile_TheMessageNamesTheKeyTheValueAndTheConventionalFile()
    {
        var found = ProjectDataFiles.Resolve(DatasetRole.Test, Declaring(test: "gone.csv"), _root, _discovery);

        Assert.Null(found.Path);
        var message = found.NotFoundMessage(DatasetRole.Test, "test data");
        Assert.Contains("data.test", message, StringComparison.Ordinal);
        Assert.Contains("'gone.csv'", message, StringComparison.Ordinal);
        Assert.Contains("datasets/test.csv", message, StringComparison.Ordinal);
    }

    [Fact]
    public void WithNothingDeclared_TheMessageIsTheConventionalOne()
    {
        var found = ProjectDataFiles.Resolve(DatasetRole.Predict, new MLoopConfig(), _root, _discovery);

        Assert.Null(found.Path);
        Assert.Equal("No data file specified and datasets/predict.csv not found.", found.NotFoundMessage(DatasetRole.Predict, "data file"));
    }

    [Theory]
    [InlineData(DatasetRole.Test, "test.csv")]
    [InlineData(DatasetRole.Predict, "predict.csv")]
    public void AProjectWithoutATrainCsv_StillFindsItsTestAndPredictFiles(DatasetRole role, string file)
    {
        var path = Write("datasets/" + file);

        Assert.Equal(path, ProjectDataFiles.Resolve(role, null, _root, _discovery).Path);
        Assert.Null(ProjectDataFiles.Resolve(DatasetRole.Train, null, _root, _discovery).Path);
    }

    [Fact]
    public void ARelativeDeclaredPath_IsRelativeToTheProjectRoot_AndAnAbsoluteOneIsItself()
    {
        var inside = Write("sets/a.csv");
        var outside = Path.Combine(Path.GetTempPath(), "mloop-data-files-outside-" + Guid.NewGuid().ToString("N") + ".csv");
        File.WriteAllText(outside, "a\n1\n");
        try
        {
            Assert.Equal(inside, ProjectDataFiles.Resolve(DatasetRole.Train, Declaring(train: "sets/a.csv"), _root, _discovery).Path);
            Assert.Equal(outside, ProjectDataFiles.Resolve(DatasetRole.Train, Declaring(train: outside), _root, _discovery).Path);
        }
        finally { File.Delete(outside); }
    }

    [Fact]
    public void FindFile_DoesNotNeedTheTrainCsv_WhereFindDatasetsDoes()
    {
        var test = Write("datasets/test.csv");

        Assert.Equal(test, _discovery.FindFile(_root, DatasetRole.Test));
        Assert.Null(_discovery.FindDatasets(_root));
    }
}
