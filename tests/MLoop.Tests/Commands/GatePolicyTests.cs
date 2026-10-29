using System.Globalization;
using System.Text;
using System.Text.Json;
using Microsoft.ML;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Storage;

namespace MLoop.Tests.Commands;

/// <summary>
/// A model's <c>gate:</c> policy in <c>mloop.yaml</c> holds back a promotion the fixed gate would allow —
/// through <c>train</c>, <c>promote</c> and its history — and refuses when it cannot be judged.
/// </summary>
[Collection("FileSystem")]
public sealed class GatePolicyTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-gate-" + Guid.NewGuid().ToString("N"));

    public GatePolicyTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    /// <summary>x decides y, with a fifth of the labels flipped — no model answers every row right.</summary>
    private static string NoisyBinary(int rows, int seed)
    {
        var csv = new StringBuilder("x,z,y\n");
        var rng = new Random(seed);
        for (var i = 0; i < rows; i++)
        {
            var x = rng.NextDouble();
            var y = x > 0.5;
            if (rng.NextDouble() < 0.2) y = !y;
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{rng.NextDouble():F4},{(y ? "yes" : "no")}\n");
        }
        return csv.ToString();
    }

    private static async Task<JsonElement> HistoryAsync()
    {
        var (_, stdout, _) = await CliRunner.RunAsync("promote", "--history", "--json");
        return JsonDocument.Parse(stdout).RootElement.GetProperty("promotions");
    }

    [Fact]
    public async Task AClassLimit_HoldsBackTrainAndPromote_UntilForced_AndTheHistorySaysSo()
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "binary-classification")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", NoisyBinary(240, 3));
        await File.WriteAllTextAsync("datasets/recheck.csv", NoisyBinary(200, 9));

        // No 'yes' may be answered 'no' on the recheck set — impossible with flipped labels.
        // init writes the platform's line endings; the edits below are made on \n.
        var yaml = (await File.ReadAllTextAsync("mloop.yaml")).Replace("\r\n", "\n").Replace("    label: Label\n", "    label: y\n", StringComparison.Ordinal);
        Assert.Contains("    task: binary-classification\n", yaml);
        yaml = yaml.Replace("    task: binary-classification\n", """
                task: binary-classification
                gate:
                  data: datasets/recheck.csv
                  limits:
                    - actual: [yes]
                      predicted: [no]
                      max: 0

            """.Replace("\r", ""), StringComparison.Ordinal);
        await File.WriteAllTextAsync("mloop.yaml", yaml);
        var (validateExit, validateOut, validateErr) = await CliRunner.RunAsync("validate");
        Assert.True(validateExit == 0, validateErr + "\n---\n" + validateOut);

        // A first model would be promoted on its metrics alone; the policy holds it in staging and says by what.
        var (trainExit, trainOut, trainErr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "60", "--max-trials", "10");
        Assert.True(trainExit == 0, trainErr + "\n---\n" + trainOut);
        Assert.Contains("gate policy refuses it", trainOut + trainErr);
        Assert.Equal(0, (await HistoryAsync()).GetArrayLength());

        var (refusedExit, refusedOut, _) = await CliRunner.RunAsync("promote", "exp-001", "--json");
        Assert.Equal(1, refusedExit);
        var error = JsonDocument.Parse(refusedOut).RootElement.GetProperty("error").GetString();
        Assert.Contains("does not pass the gate policy", error);
        Assert.Contains("of the rows labeled yes", error);
        Assert.Contains("were answered no", error);

        var (forcedExit, forcedOut, forcedErr) = await CliRunner.RunAsync("promote", "exp-001", "--json", "--force");
        Assert.True(forcedExit == 0, forcedOut + forcedErr);
        var record = Assert.Single((await HistoryAsync()).EnumerateArray());
        Assert.Equal("forced", record.GetProperty("gate").GetString());
        Assert.Contains("of the rows labeled yes", record.GetProperty("gateDetail").GetString());
    }

    [Fact]
    public async Task APolicyWhoseEvaluationSetIsMissing_Refuses_AndValidateSaysSo()
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "binary-classification")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", NoisyBinary(240, 3));
        Assert.Equal(0, (await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "60", "--max-trials", "10", "--no-promote")).ExitCode);
        var yaml = (await File.ReadAllTextAsync("mloop.yaml")).Replace("\r\n", "\n").Replace("    task: binary-classification\n", """
                task: binary-classification
                gate:
                  data: datasets/gone.csv
                  no_worse_than_production: true

            """.Replace("\r", ""), StringComparison.Ordinal);
        await File.WriteAllTextAsync("mloop.yaml", yaml);

        var (validateExit, validateOut, _) = await CliRunner.RunAsync("validate", "--json");
        var (promoteExit, promoteOut, _) = await CliRunner.RunAsync("promote", "exp-001", "--json");

        Assert.Equal(1, validateExit);
        Assert.Contains("gate.data", validateOut);
        Assert.Equal(1, promoteExit);
        Assert.Contains("evaluation set was not found", promoteOut);
    }

    // ---- no_worse_than_production, on deterministic regression models ----

    private static void WriteLinear(string path, int rows, int seed)
    {
        var random = new Random(seed);
        var lines = new List<string> { "X1,X2,Y" };
        for (var i = 0; i < rows; i++)
        {
            var x1 = random.NextDouble() * 10;
            var x2 = random.NextDouble() * 10;
            lines.Add(string.Create(CultureInfo.InvariantCulture, $"{x1:F4},{x2:F4},{2 * x1 + 3 * x2 + 1:F4}"));
        }
        File.WriteAllLines(path, lines);
    }

    private static async Task SeedRegressionAsync(string experimentId, bool scrambled, double recordedRSquared = 0.95)
    {
        var train = Path.GetFullPath($"train-{experimentId}.csv");
        WriteLinear(train, 60, 42);
        if (scrambled)
        {
            var lines = File.ReadAllLines(train);
            var labels = lines.Skip(1).Select(l => l[(l.LastIndexOf(',') + 1)..]).ToList();
            var random = new Random(5);
            labels = labels.OrderBy(_ => random.Next()).ToList();
            File.WriteAllLines(train, lines.Take(1).Concat(lines.Skip(1).Select((l, i) => l[..(l.LastIndexOf(',') + 1)] + labels[i])));
        }
        var ml = new MLContext(seed: 42);
        var inference = ColumnLayoutInference.Infer(ml, train, "Y").Options;
        var model = ml.Regression.Trainers.SingleThreadSdca(labelColumnName: "Y", featureColumnName: "Features")
            .Fit(ml.Data.CreateTextLoader(inference).Load(train));

        var fileSystem = new FileSystemManager();
        var store = new ExperimentStore(fileSystem, new ProjectDiscovery(fileSystem));
        var path = store.GetExperimentPath("default", experimentId);
        Directory.CreateDirectory(path);
        ml.Model.Save(model, null, Path.Combine(path, ExperimentLayout.ModelFileName));
        await store.SaveAsync("default", new ExperimentData
        {
            ModelName = "default",
            ExperimentId = experimentId,
            Timestamp = DateTime.UtcNow,
            Status = "Completed",
            Task = "regression",
            Config = new ExperimentConfig
            {
                DataFile = train,
                LabelColumn = "Y",
                Metric = "r_squared",
                TimeLimitSeconds = 10,
                TestSplit = 0.2,
                InputSchema = new InputSchemaInfo
                {
                    CapturedAt = DateTime.UtcNow,
                    Columns =
                    [
                        new ColumnSchema { Name = "X1", DataType = SchemaDataTypes.Numeric, Purpose = "Feature" },
                        new ColumnSchema { Name = "X2", DataType = SchemaDataTypes.Numeric, Purpose = "Feature" },
                        new ColumnSchema { Name = "Y", DataType = SchemaDataTypes.Numeric, Purpose = "Label" },
                    ],
                },
            },
            // The recorded metric says the scrambled model is fine — only the gate data can tell.
            Metrics = new Dictionary<string, double> { ["r_squared"] = recordedRSquared },
        }, CancellationToken.None);
    }

    [Fact]
    public async Task NoWorseThanProduction_RefusesAModelProductionBeatsOnTheGateData()
    {
        Directory.CreateDirectory(".mloop");
        await File.WriteAllTextAsync("mloop.yaml", """
            project: p
            models:
              default:
                task: regression
                label: Y
                gate:
                  data: recheck.csv
                  no_worse_than_production: true
            """.Replace("\r", ""));
        WriteLinear("recheck.csv", 40, 7);
        await SeedRegressionAsync("exp-001", scrambled: false);
        await SeedRegressionAsync("exp-002", scrambled: true);
        Assert.Equal(0, (await CliRunner.RunAsync("promote", "exp-001", "--json")).ExitCode);   // first model: nothing to be worse than

        var (exit, stdout, _) = await CliRunner.RunAsync("promote", "exp-002", "--json");

        Assert.Equal(1, exit);
        var error = JsonDocument.Parse(stdout).RootElement.GetProperty("error").GetString();
        Assert.Contains("production (exp-001) is better on the gate data", error);
        Assert.Contains("squared_error", error);
    }
    // ---- the recorded scores of two models trained on different data are not the comparison ----

    private static ModelRegistry NewRegistry()
    {
        var fileSystem = new FileSystemManager();
        var discovery = new ProjectDiscovery(fileSystem);
        return new ModelRegistry(fileSystem, discovery, new ExperimentStore(fileSystem, discovery));
    }

    [Fact]
    public async Task WithAPairedGate_ACandidateThatFitsTheGateDataBetter_IsNotHeldBackByALowerRecordedScore()
    {
        // Production's recorded score (0.95) is from the data it was trained on; the candidate's (0.80) from the data it
        // was trained on. On the gate data — the rows the service sees now — the candidate is the one that fits.
        Directory.CreateDirectory(".mloop");
        await File.WriteAllTextAsync("mloop.yaml", """
            project: p
            models:
              default:
                task: regression
                label: Y
                gate:
                  data: recheck.csv
                  no_worse_than_production: true
            """.Replace("\r", ""));
        WriteLinear("recheck.csv", 40, 7);
        await SeedRegressionAsync("exp-001", scrambled: true, recordedRSquared: 0.95);
        await SeedRegressionAsync("exp-002", scrambled: false, recordedRSquared: 0.80);
        Assert.Equal(0, (await CliRunner.RunAsync("promote", "exp-001", "--json")).ExitCode);
        var registry = NewRegistry();

        Assert.True(await registry.ShouldPromoteOnMetricsAsync("default", "exp-002", "r_squared"));
        Assert.True(await registry.ShouldPromoteAsync("default", "exp-002", "r_squared"));
        Assert.Null(await registry.KeptInStagingReasonAsync("default", "exp-002", "r_squared"));
    }

    [Fact]
    public async Task WithAPairedGate_ACandidateProductionBeatsOnTheGateData_IsStillRefused_WhateverItsRecordedScore()
    {
        Directory.CreateDirectory(".mloop");
        await File.WriteAllTextAsync("mloop.yaml", """
            project: p
            models:
              default:
                task: regression
                label: Y
                gate:
                  data: recheck.csv
                  no_worse_than_production: true
            """.Replace("\r", ""));
        WriteLinear("recheck.csv", 40, 7);
        await SeedRegressionAsync("exp-001", scrambled: false, recordedRSquared: 0.80);
        await SeedRegressionAsync("exp-002", scrambled: true, recordedRSquared: 0.99);
        Assert.Equal(0, (await CliRunner.RunAsync("promote", "exp-001", "--json")).ExitCode);
        var registry = NewRegistry();

        Assert.False(await registry.ShouldPromoteAsync("default", "exp-002", "r_squared"));
        Assert.Contains("production (exp-001) is better on the gate data", await registry.KeptInStagingReasonAsync("default", "exp-002", "r_squared"));
    }

    [Fact]
    public async Task WithoutAGate_TheRecordedScoresStillDecide_AndTheReasonSaysHowToCompareOnTheSameRows()
    {
        Directory.CreateDirectory(".mloop");
        await File.WriteAllTextAsync("mloop.yaml", "project: p\nmodels:\n  default:\n    task: regression\n    label: Y\n");
        await SeedRegressionAsync("exp-001", scrambled: true, recordedRSquared: 0.95);
        await SeedRegressionAsync("exp-002", scrambled: false, recordedRSquared: 0.80);
        Assert.Equal(0, (await CliRunner.RunAsync("promote", "exp-001", "--json")).ExitCode);
        var registry = NewRegistry();

        Assert.False(await registry.ShouldPromoteAsync("default", "exp-002", "r_squared"));
        var reason = await registry.KeptInStagingReasonAsync("default", "exp-002", "r_squared");
        Assert.StartsWith("it does not beat production (exp-001) on r_squared: 0.8 vs 0.95", reason);
        Assert.Contains("mloop evaluate exp-002 <data> --against production", reason);
        Assert.Contains("gate.no_worse_than_production", reason);
    }
}
