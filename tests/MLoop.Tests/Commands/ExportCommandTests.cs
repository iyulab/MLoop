using System.Globalization;
using System.Text;
using System.Text.Json;
using MLoop.Core.Export;
using MLoop.Core.Storage;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>mloop export onnx</c> — the export reads the model's input view the way prediction does, so
/// the manifest names exactly the columns the model reads (a weight column, for one, is not among
/// them), and a model ONNX cannot express is refused with the way out.
/// </summary>
[Collection("FileSystem")]
public sealed class ExportCommandTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-export-" + Guid.NewGuid().ToString("N"));

    public ExportCommandTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private async Task InitAsync()
    {
        var (exit, output, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
        Assert.True(exit == 0, output);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
    }

    [Fact]
    public async Task Export_WritesTheGraphAndAManifestOfTheColumnsTheModelReads()
    {
        await InitAsync();
        var csv = new StringBuilder("x,z,w,y\n");
        var rng = new Random(3);
        for (var i = 0; i < 200; i++)
        {
            var x = rng.NextDouble();
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{rng.NextDouble():F4},{(x > 0.5 ? 3 : 1)},{(x + rng.NextDouble() * 0.4 > 0.7 ? "yes" : "no")}\n");
        }
        await File.WriteAllTextAsync("datasets/train.csv", csv.ToString());
        var (trainExit, trainOutput, _) = await CliRunner.RunAsync(
            "train", "datasets/train.csv", "--label", "y", "--weight", "w", "--time", "5");
        Assert.True(trainExit == 0, trainOutput);

        // No experiment named: the production model.
        var (exit, output, _) = await CliRunner.RunAsync("export", "onnx", "--json");

        Assert.True(exit == 0, output);
        using var doc = JsonDocument.Parse(output);
        var onnxPath = doc.RootElement.GetProperty("onnxPath").GetString()!;
        var manifestPath = doc.RootElement.GetProperty("manifestPath").GetString()!;
        Assert.Equal(ExperimentLayout.OnnxModelFileName, Path.GetFileName(onnxPath));
        Assert.Equal(ExperimentLayout.OnnxManifestFileName, Path.GetFileName(manifestPath));
        Assert.True(new FileInfo(onnxPath).Length > 0);

        var manifest = OnnxManifest.FromJson(await File.ReadAllTextAsync(manifestPath));
        Assert.Equal("exp-001", manifest.ExperimentId);
        Assert.Equal(OnnxManifest.Sha256Of(onnxPath), manifest.OnnxSha256);
        var input = Assert.Single(manifest.Inputs);
        Assert.Equal(["x", "z"], input.SourceColumns); // the weight is not a feature
        Assert.Empty(manifest.UnusedInputs);           // and a binary model needs no label
        Assert.Equal("w", manifest.WeightColumn);
        Assert.Equal(["no", "yes"], manifest.Label.Classes);
        Assert.Contains(manifest.Outputs, o => o.Column == "Probability");
        Assert.NotNull(manifest.Metrics);
    }

    [Fact]
    public async Task Export_OfAModelWithACategoryColumn_OneHotsItAndExports()
    {
        await InitAsync();
        var csv = new StringBuilder("x,level,y\n");
        var rng = new Random(5);
        string[] levels = ["INFO", "WARN", "ERROR"];
        for (var i = 0; i < 150; i++)
        {
            var x = rng.NextDouble();
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{levels[i % 3]},{(x > 0.5 ? "yes" : "no")}\n");
        }
        await File.WriteAllTextAsync("datasets/train.csv", csv.ToString());
        var (trainExit, trainOutput, _) = await CliRunner.RunAsync(
            "train", "datasets/train.csv", "--label", "y", "--time", "5", "--no-promote");
        Assert.True(trainExit == 0, trainOutput);
        Assert.Contains("Categorical column(s): level", trainOutput);

        var (exit, output, error) = await CliRunner.RunAsync("export", "onnx", "exp-001", "--json");

        Assert.True(exit == 0, output + error);
        using var doc = JsonDocument.Parse(output);
        var manifest = OnnxManifest.FromJson(await File.ReadAllTextAsync(doc.RootElement.GetProperty("manifestPath").GetString()!));
        var level = Assert.Single(manifest.Inputs, i => i.Name == "level");
        Assert.Equal("string", level.ElementType);
    }

    [Fact]
    public async Task Export_OfAModelWithAFreeTextColumn_IsRefusedAndNamesTheColumn()
    {
        await InitAsync();
        var csv = new StringBuilder("x,note,y\n");
        var rng = new Random(5);
        string[] notes =
        [
            "the disk on the second volume failed during the nightly backup",
            "a user signed in from a new device and was asked to confirm",
            "the cache was refreshed after the configuration changed",
        ];
        for (var i = 0; i < 150; i++)
        {
            var x = rng.NextDouble();
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},\"{notes[i % 3]} {i}\",{(x > 0.5 ? "yes" : "no")}\n");
        }
        await File.WriteAllTextAsync("datasets/train.csv", csv.ToString());
        var (trainExit, trainOutput, _) = await CliRunner.RunAsync(
            "train", "datasets/train.csv", "--label", "y", "--time", "5", "--no-promote");
        Assert.True(trainExit == 0, trainOutput);

        var (exit, output, error) = await CliRunner.RunAsync("export", "onnx", "exp-001");

        Assert.Equal(1, exit);
        var said = output + error;
        Assert.Contains("'note'", said);
        Assert.Contains("type: categorical", said);
        Assert.False(File.Exists(Path.Combine("models", "default", "staging", "exp-001", ExperimentLayout.OnnxModelFileName)));
    }

    [Fact]
    public async Task Export_OfAPortableModelWithAFreeTextColumn_TakesTheRawText()
    {
        await InitAsync();
        var csv = new StringBuilder("x,note,y\n");
        var rng = new Random(7);
        string[] good = ["backup finished and every volume checked out", "the user signed in from the usual device"];
        string[] bad = ["the disk on the second volume failed during the nightly run", "a login was refused from an unknown address"];
        for (var i = 0; i < 160; i++)
        {
            var isBad = i % 2 == 0;
            var pool = isBad ? bad : good;
            csv.Append(CultureInfo.InvariantCulture, $"{rng.NextDouble():F4},\"{pool[i % 2]} {i}\",{(isBad ? "yes" : "no")}\n");
        }
        await File.WriteAllTextAsync("datasets/train.csv", csv.ToString());
        var (trainExit, trainOutput, _) = await CliRunner.RunAsync(
            "train", "datasets/train.csv", "--label", "y", "--time", "5", "--no-promote", "--portable");
        Assert.True(trainExit == 0, trainOutput);

        var (exit, output, error) = await CliRunner.RunAsync("export", "onnx", "exp-001", "--json");

        Assert.True(exit == 0, output + error);
        using var doc = JsonDocument.Parse(output);
        var manifest = OnnxManifest.FromJson(await File.ReadAllTextAsync(doc.RootElement.GetProperty("manifestPath").GetString()!));
        var note = Assert.Single(manifest.Inputs, i => i.Name == "note");
        Assert.Equal("string", note.ElementType);
        Assert.Contains("\"portable\": true", await File.ReadAllTextAsync(Path.Combine("models", "default", "staging", "exp-001", "config.json")));
    }

    [Fact]
    public async Task Portable_OnATaskItDoesNotCover_IsRefusedBeforeTraining()
    {
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "c", "--task", "clustering");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "c"));
        await File.WriteAllTextAsync("datasets/train.csv", "a,b\n1,2\n3,4\n5,6\n");

        var (exit, output, error) = await CliRunner.RunAsync("train", "datasets/train.csv", "--portable", "--time", "5");

        Assert.Equal(1, exit);
        Assert.Contains("portable", output + error);
        Assert.False(Directory.Exists(Path.Combine("models", "default", "staging", "exp-001")));
    }

    [Fact]
    public async Task Export_WithoutAProductionModel_AsksForAnExperiment()
    {
        await InitAsync();

        var (exit, output, error) = await CliRunner.RunAsync("export", "onnx");

        Assert.Equal(1, exit);
        Assert.Contains("no production model", output + error);
    }
}
