using MLoop.Core.Storage;

namespace MLoop.Core.Tests.Storage;

/// <summary>
/// Where the experiment record of a model file is. The record (the columns a model was trained on, its label and task) is beside the
/// model in an experiment folder, and is not in the <c>production</c> slot, which holds a pointer to it. Predicting with a model file named
/// by its path looked the record up by the name of the file's folder in the current project: a copy of an experiment folder, and the
/// file in the production slot, were scored without it.
/// </summary>
public sealed class ExperimentRecordLocationTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "mloop-record-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private string Folder(params string[] parts)
    {
        var dir = Path.Combine([_root, .. parts]);
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static string Record(string directory, string? id)
    {
        File.WriteAllText(Path.Combine(directory, ExperimentLayout.ConfigFileName), "{}");
        File.WriteAllText(Path.Combine(directory, ExperimentLayout.MetadataFileName), id is null ? "{}" : $"{{\"experimentId\":\"{id}\"}}");
        var model = Path.Combine(directory, ExperimentLayout.ModelFileName);
        File.WriteAllText(model, "model");
        return model;
    }

    [Fact]
    public async Task AModelBesideItsRecord_IsFound_UnderTheIdTheRecordGivesItself()
    {
        // a copy of an experiment folder, named whatever its owner named it
        var model = Record(Folder("elsewhere", "copy"), id: "exp-004");

        var found = await ExperimentRecordLocation.FindAsync(model);

        Assert.NotNull(found);
        Assert.Equal(Path.GetDirectoryName(model), found.Directory);
        Assert.Equal("exp-004", found.ExperimentId);
    }

    [Fact]
    public async Task ARecordThatGivesNoId_IsKnownByItsFoldersName()
    {
        var model = Record(Folder("exp-007"), id: null);

        Assert.Equal("exp-007", (await ExperimentRecordLocation.FindAsync(model))!.ExperimentId);
    }

    [Fact]
    public async Task TheFileInTheProductionSlot_IsFound_AsTheExperimentTheSlotPointsAt()
    {
        var staged = Folder("models", "default", ExperimentLayout.StagingDirectory, "exp-002");
        Record(staged, "exp-002");
        var slot = Folder("models", "default", ExperimentLayout.ProductionDirectory);
        File.WriteAllText(Path.Combine(slot, ExperimentLayout.ModelFileName), "model");
        File.WriteAllText(Path.Combine(slot, ExperimentLayout.MetadataFileName), "{\"modelName\":\"default\",\"experimentId\":\"exp-002\"}");

        var found = await ExperimentRecordLocation.FindAsync(Path.Combine(slot, ExperimentLayout.ModelFileName));

        Assert.NotNull(found);
        Assert.Equal(staged, found.Directory);
        Assert.Equal("exp-002", found.ExperimentId);
    }

    [Fact]
    public async Task AProductionSlot_WhoseExperimentIsGone_HasNoRecord()
    {
        var slot = Folder("models", "default", ExperimentLayout.ProductionDirectory);
        File.WriteAllText(Path.Combine(slot, ExperimentLayout.ModelFileName), "model");
        File.WriteAllText(Path.Combine(slot, ExperimentLayout.MetadataFileName), "{\"experimentId\":\"exp-009\"}");

        Assert.Null(await ExperimentRecordLocation.FindAsync(Path.Combine(slot, ExperimentLayout.ModelFileName)));
    }

    [Theory]
    [InlineData("../../elsewhere")]
    [InlineData("..")]
    [InlineData(".")]
    public async Task APointerThatNamesAPath_IsNotFollowed(string id)
    {
        // the slot's metadata is a file somebody may have handed over: its id names a folder of staging, never a path out of it
        Record(Folder("elsewhere"), "x");
        var slot = Folder("models", "default", ExperimentLayout.ProductionDirectory);
        File.WriteAllText(Path.Combine(slot, ExperimentLayout.ModelFileName), "model");
        File.WriteAllText(Path.Combine(slot, ExperimentLayout.MetadataFileName), $"{{\"experimentId\":\"{id}\"}}");

        Assert.Null(await ExperimentRecordLocation.FindAsync(Path.Combine(slot, ExperimentLayout.ModelFileName)));
    }

    [Fact]
    public async Task ABareModelFile_HasNoRecord()
    {
        var model = Path.Combine(Folder("bare"), ExperimentLayout.ModelFileName);
        File.WriteAllText(model, "model");

        Assert.Null(await ExperimentRecordLocation.FindAsync(model));
    }

    [Fact]
    public async Task AFolderThatIsNotAProductionSlot_IsNotFollowedToStaging()
    {
        // metadata.json with an experimentId sits in many folders; only a model's production slot points at staging
        Record(Folder("models", "default", ExperimentLayout.StagingDirectory, "exp-002"), "exp-002");
        var imposter = Folder("other", "notproduction");
        File.WriteAllText(Path.Combine(imposter, ExperimentLayout.ModelFileName), "model");
        File.WriteAllText(Path.Combine(imposter, ExperimentLayout.MetadataFileName), "{\"experimentId\":\"exp-002\"}");

        Assert.Null(await ExperimentRecordLocation.FindAsync(Path.Combine(imposter, ExperimentLayout.ModelFileName)));
    }
}
