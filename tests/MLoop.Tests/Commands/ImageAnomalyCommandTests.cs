using System.Text.Json;
using MLoop.Core.Embedding;
using MLoop.Core.ImageAnomaly;
using MLoop.Core.Storage;
using SkiaSharp;

namespace MLoop.Tests.Commands;

/// <summary>
/// An image-anomaly model end to end through the command tree: <c>train</c> on a folder of normal images keeps their patch
/// vectors as a bank beside the experiment, promotion carries it, <c>predict</c> scores pictures against it with a grid over the
/// image, <c>evaluate</c> tells normal folders from anomalous ones, and a model whose encoder is gone or replaced says so.
/// </summary>
[Collection("FileSystem")]
public sealed class ImageAnomalyCommandTests : IDisposable
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures", "embedding");
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string? _previousRoot = Environment.GetEnvironmentVariable(EncoderStore.RootVariable);
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-anomaly-cli-" + Guid.NewGuid().ToString("N"));

    // A 2 × 2 grid of 3-pixel patches over a 6-pixel crop — the four-dimensional patch graph the embedding tests use.
    private static readonly EncoderDefinition TinyPatches = new(
        "tiny", "1", Dimension: 4, ImagePreprocessing.ImageNet(shortestEdge: 8, cropSize: 6),
        InputName: "pixel_values", OutputName: "last_hidden_state", EncoderPooling.FirstToken, PatchSize: 3);

    public ImageAnomalyCommandTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
        Environment.SetEnvironmentVariable(EncoderStore.RootVariable, Path.Combine(_workspace, "encoders"));
        new EncoderStore().Add(TinyPatches, Path.Combine(Fixtures, "tiny-patch-encoder.onnx"));
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(EncoderStore.RootVariable, _previousRoot);
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    /// <summary>A 12 × 12 picture: a smooth gradient that varies a little by <paramref name="seed"/>, or — anomalous — a hard red block in it.</summary>
    private static void WriteImage(string path, int seed, bool anomalous = false)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var bitmap = new SKBitmap(12, 12);
        for (var y = 0; y < 12; y++)
            for (var x = 0; x < 12; x++)
            {
                var inBlock = anomalous && x is >= 2 and < 9 && y is >= 3 and < 10;
                bitmap.SetPixel(x, y, inBlock
                    ? new SKColor(255, 0, 0)
                    : new SKColor((byte)(100 + x * 6 + seed % 5), (byte)(110 + y * 5 + seed % 3), (byte)(120 + (x + y) * 2)));
            }
        using var data = SKImage.FromBitmap(bitmap).Encode(SKEncodedImageFormat.Png, 100);
        File.WriteAllBytes(path, data.ToArray());
    }

    private async Task<string> TrainAsync()
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "image-anomaly")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        for (var i = 0; i < 24; i++)
            WriteImage(Path.Combine("datasets", "normal", $"n{i:00}.png"), i);
        var (exit, output, error) = await CliRunner.RunAsync("train", "datasets/normal", "--task", "image-anomaly", "--encoder", "tiny");
        Assert.True(exit == 0, output + Environment.NewLine + error);
        return Path.Combine("models", "default", "staging", "exp-001", ExperimentLayout.ModelFileName);
    }

    [Fact]
    public async Task Training_KeepsTheBankBesideTheExperiment_AndPromotionCarriesIt()
    {
        var modelPath = await TrainAsync();

        var record = ImageAnomalyModel.ReadBeside(modelPath)!;
        Assert.Equal(("tiny", 24 - 2, 2), (record.Encoder.Id, record.Images, record.HeldOutImages));
        Assert.True(File.Exists(ImageAnomalyModel.BankPathBeside(modelPath)));
        Assert.Equal(record.BankPatches, PatchBank.Read(ImageAnomalyModel.BankPathBeside(modelPath)).Count);
        var production = Path.Combine("models", "default", "production");
        Assert.True(File.Exists(Path.Combine(production, ExperimentLayout.PatchBankFileName)), "the first model is promoted with its bank");
        Assert.True(File.Exists(Path.Combine(production, ExperimentLayout.ImageAnomalyFileName)));
    }

    [Fact]
    public async Task Predict_ScoresAnAnomalousPictureAboveANormalOne_WithAGridAndTheEncodersIdentity()
    {
        await TrainAsync();
        WriteImage(Path.Combine("new", "normal.png"), seed: 3);
        WriteImage(Path.Combine("new", "odd.png"), seed: 3, anomalous: true);

        var (exit, json, error) = await CliRunner.RunAsync("predict", "new", "--json");

        Assert.True(exit == 0, json + error);
        var root = JsonDocument.Parse(json).RootElement;
        Assert.Equal("tiny", root.GetProperty("encoder").GetProperty("id").GetString());
        Assert.Equal(2, root.GetProperty("grid").GetProperty("rows").GetInt32());
        var images = root.GetProperty("images").EnumerateArray().ToDictionary(
            i => Path.GetFileName(i.GetProperty("file").GetString()!), i => i);
        Assert.True(images["odd.png"].GetProperty("score").GetDouble() > images["normal.png"].GetProperty("score").GetDouble());
        Assert.Equal(4, images["odd.png"].GetProperty("distances").GetArrayLength());
        Assert.True(images["odd.png"].GetProperty("crop").GetProperty("x1").GetDouble() > images["odd.png"].GetProperty("crop").GetProperty("x0").GetDouble());
        Assert.Equal(JsonValueKind.Object, root.GetProperty("baseline").ValueKind);
    }

    [Fact]
    public async Task Predict_WritesTheScoresFile_AndNamesWhatIsMissingFromItsInput()
    {
        await TrainAsync();
        WriteImage(Path.Combine("new", "a.png"), seed: 1);

        var (exit, _, error) = await CliRunner.RunAsync("predict", "new");
        Assert.True(exit == 0, error);
        Assert.Single(Directory.GetFiles("predictions", "*-scores-*.json"));

        Assert.Contains("name an image or a folder", (await CliRunner.RunAsync("predict")).Stderr);
        Directory.CreateDirectory("empty");
        Assert.Contains("No images found", (await CliRunner.RunAsync("predict", "empty")).Stderr);
    }

    [Fact]
    public async Task Evaluate_TellsTheNormalFolderFromTheAnomalousOnes_ByAurocAndMissedAnomalies()
    {
        await TrainAsync();
        for (var i = 0; i < 12; i++)
        {
            WriteImage(Path.Combine("check", "good", $"g{i:00}.png"), 100 + i);
            WriteImage(Path.Combine("check", "scratch", $"s{i:00}.png"), 100 + i, anomalous: true);
        }

        var (exit, json, error) = await CliRunner.RunAsync("evaluate", "exp-001", "check", "--json");

        Assert.True(exit == 0, json + error);
        var root = JsonDocument.Parse(json).RootElement;
        var metrics = root.GetProperty("testMetrics");
        Assert.True(metrics.GetProperty("auroc").GetDouble() > 0.95, json);
        Assert.Equal(12, metrics.GetProperty("anomalous_images").GetInt32());
        Assert.Equal("good", root.GetProperty("normalClass")[0].GetString());

        // a folder with no normal class, or no anomalous one, cannot tell them apart
        Directory.CreateDirectory(Path.Combine("lonely", "defects"));
        WriteImage(Path.Combine("lonely", "defects", "a.png"), 1, anomalous: true);
        var lonely = await CliRunner.RunAsync("evaluate", "exp-001", "lonely");
        Assert.NotEqual(0, lonely.ExitCode);
        Assert.Contains("normal", lonely.Stderr);
        Assert.Equal(0, (await CliRunner.RunAsync("evaluate", "exp-001", "check", "--normal-class", "good")).ExitCode);
    }

    [Fact]
    public async Task AModelWhoseEncoderIsGoneOrReplaced_RefusesToScore_NamingTheEncoder()
    {
        await TrainAsync();
        WriteImage(Path.Combine("new", "a.png"), seed: 1);

        // Same id, other weights: its distances would be another space.
        var other = Path.Combine(_workspace, "other.onnx");
        var replacement = new EncoderStore().Find("tiny")!.Definition with { Sha256 = null };   // the installed one pins its weights
        File.Copy(Path.Combine(Fixtures, "tiny-encoder.onnx"), other, overwrite: true);   // another graph of the same input and output names
        new EncoderStore().Add(replacement, other);
        var (replaced, _, replacedErr) = await CliRunner.RunAsync("predict", "new");
        Assert.NotEqual(0, replaced);
        Assert.True(replacedErr.Contains("tiny") && replacedErr.Contains("other weights"), replacedErr);

        new EncoderStore().Remove("tiny");
        var (gone, _, goneErr) = await CliRunner.RunAsync("predict", "new");
        Assert.NotEqual(0, gone);
        Assert.True(goneErr.Contains("not installed"), goneErr);
    }
}
