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
}
