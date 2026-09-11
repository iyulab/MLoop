using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// Pins what an experiment records about its data: a content fingerprint that two copies of the
/// same bytes share and one changed byte breaks, written with its algorithm in front.
/// </summary>
public class DataFingerprintTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "mloop-fp-" + Guid.NewGuid().ToString("N"));

    public DataFingerprintTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private async Task<string> WriteAsync(string name, string content)
    {
        var path = Path.Combine(_dir, name);
        await File.WriteAllTextAsync(path, content);
        return path;
    }

    [Fact]
    public async Task The_same_bytes_have_the_same_fingerprint_whatever_the_file_is_called()
    {
        var a = await WriteAsync("train.csv", "x,y\n1,2\n3,4\n");
        var b = await WriteAsync("copy-elsewhere.csv", "x,y\n1,2\n3,4\n");

        Assert.Equal(await DataFingerprint.ComputeAsync(a), await DataFingerprint.ComputeAsync(b));
    }

    [Fact]
    public async Task One_changed_byte_changes_the_fingerprint()
    {
        var a = await WriteAsync("a.csv", "x,y\n1,2\n3,4\n");
        var b = await WriteAsync("b.csv", "x,y\n1,2\n3,5\n");

        Assert.NotEqual(await DataFingerprint.ComputeAsync(a), await DataFingerprint.ComputeAsync(b));
    }

    [Fact]
    public async Task A_fingerprint_names_its_algorithm_and_is_lower_case_hex()
    {
        var path = await WriteAsync("c.csv", "x\n1\n");

        var fingerprint = await DataFingerprint.ComputeAsync(path);

        Assert.StartsWith(DataFingerprint.Prefix, fingerprint);
        Assert.Equal(DataFingerprint.Prefix.Length + 64, fingerprint.Length);
        Assert.True(DataFingerprint.IsFingerprint(fingerprint));
        // The known SHA-256 of "x\n1\n" — the value is the standard algorithm's, not a private one.
        Assert.Equal(
            DataFingerprint.Prefix + Convert.ToHexStringLower(
                System.Security.Cryptography.SHA256.HashData("x\n1\n"u8.ToArray())),
            fingerprint);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("sha256:")]
    [InlineData("md5:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")]
    [InlineData("sha256:0123456789ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef")]
    [InlineData("sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcde")]
    public void A_value_of_another_shape_is_not_a_fingerprint(string? value)
    {
        Assert.False(DataFingerprint.IsFingerprint(value));
    }

    [Fact]
    public async Task A_missing_file_throws_the_ordinary_io_exception()
    {
        await Assert.ThrowsAsync<FileNotFoundException>(
            () => DataFingerprint.ComputeAsync(Path.Combine(_dir, "absent.csv")));
    }
}
