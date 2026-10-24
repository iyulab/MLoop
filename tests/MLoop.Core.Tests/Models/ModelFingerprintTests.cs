using System.Security.Cryptography;
using MLoop.Core.Models;

namespace MLoop.Core.Tests.Models;

public sealed class ModelFingerprintTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "mloop_fp_" + Guid.NewGuid().ToString("N"));

    public ModelFingerprintTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    [Fact]
    public void Sha256_is_the_files_content_hash()
    {
        var path = Path.Combine(_dir, "model.zip");
        File.WriteAllBytes(path, [1, 2, 3]);

        Assert.Equal(Convert.ToHexStringLower(SHA256.HashData(new byte[] { 1, 2, 3 })), ModelFingerprint.Sha256(path));
    }

    [Fact]
    public void Sha256_follows_a_model_replaced_in_place()
    {
        // Promotion overwrites production/model.zip; a remembered hash must not outlive the file it was taken of.
        var path = Path.Combine(_dir, "model.zip");
        File.WriteAllBytes(path, [1, 2, 3]);
        var first = ModelFingerprint.Sha256(path);

        File.WriteAllBytes(path, [4, 5, 6, 7]);
        File.SetLastWriteTimeUtc(path, DateTime.UtcNow.AddMinutes(1));

        Assert.NotEqual(first, ModelFingerprint.Sha256(path));
        Assert.Equal(Convert.ToHexStringLower(SHA256.HashData(new byte[] { 4, 5, 6, 7 })), ModelFingerprint.Sha256(path));
    }
}
