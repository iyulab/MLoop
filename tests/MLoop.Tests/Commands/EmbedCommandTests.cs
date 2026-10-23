using System.Text.Json;
using MLoop.Core.Embedding;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>mloop encoder add · list · remove</c> and <c>mloop embed</c> through the real command tree, against an encoder
/// store in a temp directory (<see cref="EncoderStore.RootVariable"/>) — no project needed.
/// </summary>
[Collection("FileSystem")]
public sealed class EmbedCommandTests : IDisposable
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures", "embedding");
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"mloop-embed-{Guid.NewGuid():N}");
    private readonly string? _previousRoot = Environment.GetEnvironmentVariable(EncoderStore.RootVariable);

    public EmbedCommandTests()
    {
        Directory.CreateDirectory(_dir);
        Environment.SetEnvironmentVariable(EncoderStore.RootVariable, Path.Combine(_dir, "encoders"));
        File.WriteAllText(Path.Combine(_dir, "encoder.json"), """
            {
              "id": "tiny", "version": "1", "dimension": 4,
              "preprocessing": { "shortestEdge": 8, "cropSize": 6, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] },
              "inputName": "pixel_values", "outputName": "last_hidden_state", "pooling": "first-token"
            }
            """);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(EncoderStore.RootVariable, _previousRoot);
        if (Directory.Exists(_dir))
            Directory.Delete(_dir, recursive: true);
    }

    private Task<(int ExitCode, string Stdout, string Stderr)> AddTiny() =>
        CliRunner.RunAsync("encoder", "add", Path.Combine(_dir, "encoder.json"), "--model", Path.Combine(Fixtures, "tiny-encoder.onnx"));

    [Fact]
    public async Task AnAddedEncoder_IsListed_AndEmbedsAnImage()
    {
        var (added, _, addErr) = await AddTiny();
        Assert.True(added == 0, addErr);

        var (listed, list, _) = await CliRunner.RunAsync("encoder", "list", "--json");
        Assert.Equal(0, listed);
        using (var doc = JsonDocument.Parse(list))
            Assert.Equal("tiny", doc.RootElement.GetProperty("encoders")[0].GetProperty("id").GetString());

        var (embedded, json, embedErr) = await CliRunner.RunAsync("embed", Path.Combine(Fixtures, "pattern.png"), "--json");
        Assert.True(embedded == 0, embedErr);
        using var result = JsonDocument.Parse(json);
        var expected = JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures, "expected.json"))).RootElement.GetProperty("embedding");
        var vector = result.RootElement.GetProperty("vectors")[0];
        for (var d = 0; d < 4; d++)
            Assert.Equal(expected[d].GetSingle(), vector[d].GetSingle(), 4);
        Assert.Equal(64, result.RootElement.GetProperty("encoder").GetProperty("sha256").GetString()!.Length);
    }

    [Fact]
    public async Task EmbedWithoutAnEncoder_SaysHowToAddOne()
    {
        var (exit, _, stderr) = await CliRunner.RunAsync("embed", Path.Combine(Fixtures, "pattern.png"));

        Assert.Equal(1, exit);
        Assert.Contains("mloop encoder add", stderr);
    }

    [Fact]
    public async Task ARemovedEncoder_IsNoLongerListed()
    {
        await AddTiny();

        var (removed, _, _) = await CliRunner.RunAsync("encoder", "remove", "tiny");
        var (_, list, _) = await CliRunner.RunAsync("encoder", "list", "--json");

        Assert.Equal(0, removed);
        using var doc = JsonDocument.Parse(list);
        Assert.Equal(0, doc.RootElement.GetProperty("encoders").GetArrayLength());
    }
}
