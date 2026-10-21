using System.Globalization;
using System.Text;
using MLoop.CLI.Commands;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>train --weight</c> / <c>weight:</c> — the weight column weights the rows and is never a feature;
/// a weight the run could not honour is refused before training, not silently ignored.
/// </summary>
[Collection("FileSystem")]
public sealed class TrainWeightTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-weight-" + Guid.NewGuid().ToString("N"));

    public TrainWeightTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private static ModelDefinition Definition(string task = "binary-classification", string? weight = "w",
        Dictionary<string, ColumnOverride>? columns = null)
        => new() { Task = task, Label = "y", Weight = weight, Columns = columns };

    [Fact]
    public void AWeightTheRunCanHonour_HasNoConflict()
        => Assert.Null(ModelTrainingConfig.WeightConflict(Definition(), balance: null));

    [Theory]
    [InlineData("clustering", "w", null, "applies to binary-classification")]
    [InlineData("binary-classification", "y", null, "is the label column")]
    [InlineData("binary-classification", "w", "auto", "--balance")]
    public void AWeightTheRunCouldNotHonour_IsRefusedWithTheReason(string task, string weight, string? balance, string expected)
        => Assert.Contains(expected, ModelTrainingConfig.WeightConflict(Definition(task, weight), balance));

    [Fact]
    public void AWeightDeclaredAsAFeature_IsRefused()
    {
        var definition = Definition(columns: new() { ["W"] = new ColumnOverride { Type = "numeric" } });

        Assert.Contains("never a feature", ModelTrainingConfig.WeightConflict(definition, balance: "none"));
    }

    [Fact]
    public async Task Train_WithAWeight_RecordsTheWeightAsExcluded()
    {
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));

        var csv = new StringBuilder("x,z,w,y\n");
        var rng = new Random(3);
        for (var i = 0; i < 200; i++)
        {
            var x = rng.NextDouble();
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{rng.NextDouble():F4},{(x > 0.5 ? 3 : 1)},{(x + rng.NextDouble() * 0.4 > 0.7 ? "true" : "false")}\n");
        }
        await File.WriteAllTextAsync("datasets/train.csv", csv.ToString());

        var (exit, output, _) = await CliRunner.RunAsync(
            "train", "datasets/train.csv", "--label", "y", "--weight", "w", "--time", "5", "--no-promote");

        Assert.True(exit == 0, output);
        var fileSystem = new FileSystemManager();
        var store = new ExperimentStore(fileSystem, new ProjectDiscovery(fileSystem));
        var experiment = await store.LoadAsync("default", "exp-001", CancellationToken.None);
        var w = experiment.Config!.InputSchema!.Columns.Single(c => c.Name == "w");
        Assert.Equal("Exclude", w.Purpose);
        Assert.Equal("w", experiment.Config.WeightColumn);

        // evaluate counts the rows by the experiment's weight when the file carries it…
        var (evalExit, evalJson, _) = await CliRunner.RunAsync("evaluate", "exp-001", "datasets/train.csv", "--json");
        Assert.True(evalExit == 0, evalJson);
        Assert.Equal("w", System.Text.Json.JsonDocument.Parse(evalJson).RootElement.GetProperty("weightColumn").GetString());
        // …and so does its confusion matrix: the classes in the label's own words, each cell a weighted sum.
        var confusion = System.Text.Json.JsonDocument.Parse(evalJson).RootElement.GetProperty("confusion");
        Assert.Equal(["false", "true"], confusion.GetProperty("classes").EnumerateArray().Select(c => c.GetString()));
        var weightSum = (await File.ReadAllLinesAsync("datasets/train.csv")).Skip(1).Sum(l => double.Parse(l.Split(',')[2], CultureInfo.InvariantCulture));
        Assert.Equal(weightSum, confusion.GetProperty("total").GetDouble(), 6);

        // …and says so when it does not, rather than comparing a weighted training column with an unweighted test one.
        var lines = (await File.ReadAllLinesAsync("datasets/train.csv")).Select(l => string.Join(',', l.Split(',').Where((_, i) => i != 2)));
        await File.WriteAllLinesAsync("datasets/no-weight.csv", lines);
        var (plainExit, plainJson, plainErr) = await CliRunner.RunAsync("evaluate", "exp-001", "datasets/no-weight.csv", "--json");
        Assert.True(plainExit == 0, plainJson + plainErr);
        Assert.Equal(System.Text.Json.JsonValueKind.Null,
            System.Text.Json.JsonDocument.Parse(plainJson).RootElement.GetProperty("weightColumn").ValueKind);
        Assert.Contains("weighted by 'w'", plainErr);
    }

    [Fact]
    public async Task Train_WithAWeightOnAnUnsupportedTask_FailsBeforeTraining()
    {
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "clustering");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", "a,b,w\n1,2,1\n2,3,1\n3,4,2\n");

        var (exit, _, stderr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--weight", "w", "--time", "5");

        Assert.Equal(1, exit);
        Assert.Contains("weighted training applies to", stderr);
    }

    [Fact]
    public async Task AWeightedModel_IsNotPromotedOverAnUnweightedOne_OnMetricsOfAnotherDistribution()
    {
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        var csv = new StringBuilder("x,z,w,y\n");
        var rng = new Random(9);
        for (var i = 0; i < 200; i++)
        {
            var x = rng.NextDouble();
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{rng.NextDouble():F4},{(x > 0.5 ? 3 : 1)},{(x + rng.NextDouble() * 0.4 > 0.7 ? "true" : "false")}\n");
        }
        await File.WriteAllTextAsync("datasets/train.csv", csv.ToString());
        await File.WriteAllTextAsync("mloop.yaml",
            "project: p\nmodels:\n  default:\n    task: binary-classification\n    label: y\n    columns:\n      w:\n        type: ignore\n");

        // On failure the cause is on stderr (errors go there); stdout alone ends mid-narration.
        var (first, firstOut, firstErr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "5");
        Assert.True(first == 0, firstErr + "\n---\n" + firstOut);
        var (second, secondOut, secondErr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--weight", "w", "--time", "5");
        Assert.True(second == 0, secondErr + "\n---\n" + secondOut);

        Assert.Contains("Not compared with production", secondOut + secondErr);
        var fileSystem = new FileSystemManager();
        var discovery = new ProjectDiscovery(fileSystem);
        var registry = new ModelRegistry(fileSystem, discovery, new ExperimentStore(fileSystem, discovery));
        Assert.Equal("exp-001", (await registry.GetProductionAsync("default"))!.ExperimentId);

        var (cmpExit, cmpJson, _) = await CliRunner.RunAsync("compare", "exp-001", "exp-002", "--json");
        Assert.True(cmpExit == 0, cmpJson);
        using var doc = System.Text.Json.JsonDocument.Parse(cmpJson);
        Assert.True(doc.RootElement.GetProperty("weightsDiffer").GetBoolean());
        var byId = doc.RootElement.GetProperty("experiments").EnumerateArray().ToDictionary(e => e.GetProperty("id").GetString()!);
        Assert.Equal(System.Text.Json.JsonValueKind.Null, byId["exp-001"].GetProperty("weightColumn").ValueKind);
        Assert.Equal("w", byId["exp-002"].GetProperty("weightColumn").GetString());
    }
}
