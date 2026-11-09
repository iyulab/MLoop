using MLoop.Core.Models;

namespace MLoop.Core.Tests.Models;

public sealed class ModelIntegrityTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "mloop_integrity_" + Guid.NewGuid().ToString("N"));

    public ModelIntegrityTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
    }

    /// <summary>A slot as a promotion leaves it: the model and the pointer that holds the model's hash (null = a pointer from before the hash was recorded).</summary>
    private string Slot(string directory, byte[] model, bool recordHash = true, string? pointer = null)
    {
        var dir = Path.Combine(_root, directory);
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, "model.zip");
        File.WriteAllBytes(path, model);
        File.WriteAllText(Path.Combine(dir, "metadata.json"),
            pointer ?? (recordHash
                ? $"{{\"experimentId\":\"exp-001\",\"modelSha256\":\"{ModelFingerprint.Sha256(path)}\"}}"
                : "{\"experimentId\":\"exp-001\"}"));
        return path;
    }

    [Fact]
    public void A_production_model_that_is_the_file_its_promotion_recorded_is_not_questioned()
    {
        Assert.Null(ModelIntegrity.Mismatch(Slot("production", [1, 2, 3])));
    }

    [Fact]
    public void A_production_model_changed_after_its_promotion_is_said_so_with_both_hashes_and_the_way_back()
    {
        var path = Slot("production", [1, 2, 3]);
        var recorded = ModelFingerprint.Sha256(path);
        File.WriteAllBytes(path, [9, 9, 9, 9, 9]);   // another length: the remembered hash of the old file must not be reused

        var message = ModelIntegrity.Mismatch(path);

        Assert.NotNull(message);
        Assert.Contains("changed after it was promoted", message, StringComparison.Ordinal);
        Assert.Contains(recorded[..12], message, StringComparison.Ordinal);
        Assert.Contains(ModelFingerprint.Sha256(path)[..12], message, StringComparison.Ordinal);
        Assert.Contains("mloop rollback", message, StringComparison.Ordinal);
    }

    [Fact]
    public void A_slot_promoted_before_the_hash_was_recorded_has_nothing_to_compare()
    {
        Assert.Null(ModelIntegrity.Mismatch(Slot("production", [1, 2, 3], recordHash: false)));
    }

    [Fact]
    public void Only_a_production_slots_model_is_compared_an_experiments_own_metadata_is_not_a_promotion_record()
    {
        var staged = Slot(Path.Combine("staging", "exp-001"), [1, 2, 3], pointer: "{\"modelSha256\":\"0000\"}");
        Assert.Null(ModelIntegrity.Mismatch(staged));
        var elsewhere = Slot("somewhere", [1, 2, 3], pointer: "{\"modelSha256\":\"0000\"}");
        Assert.Null(ModelIntegrity.Mismatch(elsewhere));
    }

    [Theory]
    [InlineData("not json at all")]
    [InlineData("")]
    public void A_pointer_that_cannot_be_read_never_breaks_the_load(string pointer)
    {
        Assert.Null(ModelIntegrity.Mismatch(Slot("production", [1, 2, 3], pointer: pointer)));
    }

    [Fact]
    public void A_missing_model_or_pointer_reads_as_nothing_to_compare()
    {
        Assert.Null(ModelIntegrity.Mismatch(Path.Combine(_root, "production", "model.zip")));
        var path = Slot("production", [1, 2, 3]);
        File.Delete(Path.Combine(_root, "production", "metadata.json"));
        Assert.Null(ModelIntegrity.Mismatch(path));
    }

    private void ProjectFile(string yaml) => File.WriteAllText(Path.Combine(_root, "mloop.yaml"), yaml);

    private string ChangedProductionModel()
    {
        var path = Slot("production", [1, 2, 3]);
        File.WriteAllBytes(path, [9, 9, 9, 9, 9]);
        return path;
    }

    [Fact]
    public void A_changed_production_model_is_refused_when_the_project_asks_for_it_and_the_refusal_says_where_it_was_asked()
    {
        ProjectFile("project: demo\nintegrity:\n  refuse_changed_model: true\n");
        var path = ChangedProductionModel();

        var refusal = Assert.Throws<ChangedModelException>(() => ModelIntegrity.Enforce(path));

        Assert.Contains("changed after it was promoted", refusal.Message, StringComparison.Ordinal);
        Assert.Contains("refuse_changed_model", refusal.Message, StringComparison.Ordinal);
        Assert.Contains("mloop rollback", refusal.Message, StringComparison.Ordinal);
        Assert.Null(ModelIntegrity.Warning(path));   // said once, by the refusal - not also as a warning
    }

    [Theory]
    [InlineData(null)]                                                // no project file at all
    [InlineData("project: demo\n")]                                   // the key absent: the old behaviour
    [InlineData("integrity:\n  refuse_changed_model: false\n")]
    [InlineData("integrity: [this is not a mapping\n")]               // unreadable: a check does not turn a typo into an outage
    public void A_changed_production_model_is_only_warned_about_unless_the_project_asks_for_a_refusal(string? yaml)
    {
        if (yaml is not null)
            ProjectFile(yaml);
        var path = ChangedProductionModel();

        ModelIntegrity.Enforce(path);

        Assert.NotNull(ModelIntegrity.Warning(path));
        Assert.False(ModelIntegrity.RefusesChangedModel(path));
    }

    [Fact]
    public void A_production_model_that_matches_its_record_is_never_refused_even_when_refusal_is_on()
    {
        ProjectFile("integrity:\n  refuse_changed_model: true\n");
        var path = Slot("production", [1, 2, 3]);

        ModelIntegrity.Enforce(path);

        Assert.Null(ModelIntegrity.Warning(path));
    }

    [Fact]
    public void A_model_outside_the_production_slot_is_never_refused_a_hand_placed_model_elsewhere_is_the_users_own()
    {
        ProjectFile("integrity:\n  refuse_changed_model: true\n");
        ModelIntegrity.Enforce(Slot(Path.Combine("staging", "exp-001"), [1, 2, 3], pointer: "{\"modelSha256\":\"0000\"}"));
    }

    [Fact]
    public void Every_scoring_load_passes_the_refusal_before_the_file_is_read()
    {
        ProjectFile("integrity:\n  refuse_changed_model: true\n");
        var path = ChangedProductionModel();   // not a model archive: without the refusal this would be an UnreadableStateFileException

        Assert.Throws<ChangedModelException>(() => ScoringModel.Load(new Microsoft.ML.MLContext(seed: 1), path));
    }

    [Fact]
    public void The_project_file_is_found_above_the_slot_where_a_project_keeps_it_models_name_production()
    {
        ProjectFile("integrity:\n  refuse_changed_model: true\n");
        var slot = Path.Combine(_root, "models", "default", "production");
        Directory.CreateDirectory(slot);
        var path = Path.Combine(slot, "model.zip");
        File.WriteAllBytes(path, [1, 2, 3]);
        File.WriteAllText(Path.Combine(slot, "metadata.json"), $"{{\"experimentId\":\"exp-001\",\"modelSha256\":\"{ModelFingerprint.Sha256(path)}\"}}");
        File.WriteAllBytes(path, [9, 9, 9, 9, 9]);

        Assert.True(ModelIntegrity.RefusesChangedModel(path));
        Assert.Throws<ChangedModelException>(() => ModelIntegrity.Enforce(path));
    }
}
