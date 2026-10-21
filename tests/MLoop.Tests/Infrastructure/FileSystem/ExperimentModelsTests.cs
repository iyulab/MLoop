using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Storage;

namespace MLoop.Tests.Infrastructure.FileSystem;

[Collection("FileSystem")]
public sealed class ExperimentModelsTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _root = Path.Combine(Path.GetTempPath(), "mloop-experiment-models-" + Guid.NewGuid().ToString("N"));
    private readonly ExperimentStore _store;

    public ExperimentModelsTests()
    {
        Directory.CreateDirectory(Path.Combine(_root, ".mloop"));
        Directory.SetCurrentDirectory(_root);
        var fs = new FileSystemManager();
        _store = new ExperimentStore(fs, new ProjectDiscovery(fs));
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_root, recursive: true); } catch { /* temp */ }
    }

    private string Experiment(string id, bool withModel)
    {
        var path = _store.GetExperimentPath("default", id);
        Directory.CreateDirectory(path);
        File.WriteAllText(Path.Combine(path, ExperimentLayout.MetadataFileName), $$"""{"experimentId":"{{id}}","modelName":"default"}""");
        if (withModel)
            File.WriteAllText(Path.Combine(path, ExperimentLayout.ModelFileName), "zip");
        return path;
    }

    [Fact]
    public void ATrainedExperiment_GivesItsModel()
    {
        var path = Experiment("exp-001", withModel: true);

        var (model, refusal) = ExperimentModels.Find(_store, "default", "exp-001");

        Assert.Null(refusal);
        Assert.Equal(Path.Combine(path, ExperimentLayout.ModelFileName), model);
    }

    [Fact]
    public void AnExperimentWithoutAModel_SaysSo()
    {
        Experiment("exp-002", withModel: false);

        var (model, refusal) = ExperimentModels.Find(_store, "default", "exp-002");

        Assert.Null(model);
        Assert.Equal(ExperimentModels.NoTrainedModel("exp-002"), refusal);
    }

    [Fact]
    public void AnUnknownExperiment_IsNotFound()
    {
        var (model, refusal) = ExperimentModels.Find(_store, "default", "exp-404");

        Assert.Null(model);
        Assert.Equal(ExperimentModels.NotFound("default", "exp-404"), refusal);
    }
}
