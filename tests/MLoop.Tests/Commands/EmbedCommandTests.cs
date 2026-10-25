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

    private Task<(int ExitCode, string Stdout, string Stderr)> AddTinyText()
    {
        File.WriteAllText(Path.Combine(_dir, "text-encoder.json"), """
            {
              "id": "tiny-text", "version": "1", "dimension": 4, "kind": "text",
              "text": { "maxTokens": 12, "tokenIds": "xlm-roberta" },
              "inputName": "input_ids", "outputName": "last_hidden_state", "pooling": "mean-of-tokens"
            }
            """);
        return CliRunner.RunAsync("encoder", "add", Path.Combine(_dir, "text-encoder.json"),
            "--model", Path.Combine(Fixtures, "tiny-text-encoder.onnx"), "--tokenizer", Path.Combine(Fixtures, "tiny-tokenizer.model"));
    }

    [Fact]
    public async Task AnAddedTextEncoder_EmbedsTexts_FromOptionsAndAFile_AndCountsTheTruncated()
    {
        var (added, _, addErr) = await AddTinyText();
        Assert.True(added == 0, addErr);
        var cases = JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures, "text.json"))).RootElement.GetProperty("cases");
        var file = Path.Combine(_dir, "texts.txt");
        File.WriteAllLines(file, cases.EnumerateArray().Skip(1).Select(c => c.GetProperty("text").GetString()!).Where(t => t.Length > 0));

        var (exit, json, stderr) = await CliRunner.RunAsync("embed", "--text", cases[0].GetProperty("text").GetString()!, "--text-file", file, "--json");

        Assert.True(exit == 0, stderr);
        using var doc = JsonDocument.Parse(json);
        var vectors = doc.RootElement.GetProperty("vectors");
        Assert.Equal(cases.GetArrayLength() - 1, vectors.GetArrayLength());   // the file's empty line is skipped
        Assert.Equal(cases[0].GetProperty("vector")[0].GetSingle(), vectors[0][0].GetSingle(), 4);
        Assert.Equal("text", doc.RootElement.GetProperty("encoder").GetProperty("kind").GetString());
        Assert.Equal(cases.EnumerateArray().Count(c => c.GetProperty("truncated").GetBoolean()), doc.RootElement.GetProperty("truncated").GetInt32());
        Assert.Contains("longer than encoder 'tiny-text' reads", doc.RootElement.GetProperty("warnings")[0].GetString());
    }

    [Fact]
    public async Task ImagesAndTextsTogether_OrNothing_IsRefused()
    {
        var (both, _, bothErr) = await CliRunner.RunAsync("embed", Path.Combine(Fixtures, "pattern.png"), "--text", "a");
        var (none, _, noneErr) = await CliRunner.RunAsync("embed");

        Assert.Equal(1, both);
        Assert.Contains("not both", bothErr);
        Assert.Equal(1, none);
        Assert.Contains("Nothing to embed", noneErr);
    }

    [Fact]
    public async Task ATextEncoderAddedWithoutItsTokenizer_IsRefused()
    {
        File.WriteAllText(Path.Combine(_dir, "text-encoder.json"), """
            { "id": "t", "version": "1", "dimension": 4, "kind": "text", "text": { "maxTokens": 12 },
              "inputName": "input_ids", "outputName": "last_hidden_state", "pooling": "mean-of-tokens" }
            """);

        var (exit, _, stderr) = await CliRunner.RunAsync("encoder", "add", Path.Combine(_dir, "text-encoder.json"), "--model", Path.Combine(Fixtures, "tiny-text-encoder.onnx"));

        Assert.Equal(1, exit);
        Assert.Contains("tokenizer", stderr);
    }

    [Fact]
    public async Task EmbedWithoutAnEncoder_SaysHowToAddOne()
    {
        var (exit, _, stderr) = await CliRunner.RunAsync("embed", Path.Combine(Fixtures, "pattern.png"));

        Assert.Equal(1, exit);
        Assert.Contains("mloop encoder install", stderr);
        Assert.Contains("mloop encoder add", stderr);
    }

    // Only catalog encoders install by name; anything else is added from a file, and the refusal says so.
    [Fact]
    public async Task InstallingAnEncoderNotInTheCatalog_IsRefused_WithTheCatalogAndTheAddCommand()
    {
        var (exit, _, stderr) = await CliRunner.RunAsync("encoder", "install", "not-an-encoder");

        Assert.Equal(1, exit);
        Assert.Contains("dinov2-small", stderr);
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
