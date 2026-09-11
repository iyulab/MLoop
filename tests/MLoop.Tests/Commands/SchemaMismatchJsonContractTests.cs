using Microsoft.ML;
using Microsoft.ML.AutoML;
using MLoop.CLI;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Models;
using MLoop.Core.Storage;
using Spectre.Console;
using System.Text.Json;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>predict --json</c> and <c>evaluate --json</c> owe their consumer a document at the one exit
/// these commands exist to report clearly: the data does not match the model's schema. Before
/// 0.31.0 both rendered the mismatch to the terminal in four parts and exited non-zero with an empty
/// stdout, so a consumer parsing the output got the same failure as parsing prose.
/// </summary>
/// <remarks>
/// The check sits <b>behind a successfully loaded model</b>, which is why the existing exit-contract
/// guard never reached it — that guard points its commands at a project with no model at all and
/// exercises a different exit. A synthetic fixture that stops short of a loadable model reaches the
/// same wrong exit and passes for the wrong reason (measured, and removed, before 0.31.0 shipped).
/// So this fixture trains a real model — a direct SDCA fit, no AutoML search — stores it as an
/// experiment with a recorded input schema, and hands the commands a file that is missing a column
/// that schema requires. The negative control at the end proves the same fixture gets past the load
/// when the file <i>does</i> match, so a green result here is not another test of the wrong exit.
/// </remarks>
[Collection("FileSystem")]
public class SchemaMismatchJsonContractTests : IDisposable
{
    private const string LabelColumn = "Y";

    private readonly string _testProjectRoot;
    private readonly string _originalDirectory;
    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;
    private readonly IExperimentStore _experimentStore;

    public SchemaMismatchJsonContractTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();
        _testProjectRoot = Path.Combine(Path.GetTempPath(), "mloop-schema-mismatch-json-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_testProjectRoot);
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, ".mloop"));

        _fileSystem = new FileSystemManager();
        _projectDiscovery = new ProjectDiscovery(_fileSystem);
        Directory.SetCurrentDirectory(_testProjectRoot);
        _experimentStore = new ExperimentStore(_fileSystem, _projectDiscovery);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); }
        catch { try { Directory.SetCurrentDirectory(Path.GetTempPath()); } catch { } }
        if (Directory.Exists(_testProjectRoot))
        {
            try { Directory.Delete(_testProjectRoot, recursive: true); } catch { }
        }
    }

    private static async Task<(int ExitCode, string Stdout, string Stderr)> RunAsync(params string[] args)
    {
        var originalOut = Console.Out;
        var originalError = Console.Error;
        var originalAnsiConsole = AnsiConsole.Console;
        var buffer = new StringWriter();
        var errorBuffer = new StringWriter();
        try
        {
            Console.SetOut(buffer);
            Console.SetError(errorBuffer);
            AnsiConsole.Console = AnsiConsole.Create(new AnsiConsoleSettings { Out = new AnsiConsoleOutput(buffer) });
            var exitCode = await Program.ExecuteAsync(Program.BuildRootCommand(), args);
            return (exitCode, buffer.ToString(), errorBuffer.ToString());
        }
        finally
        {
            Console.SetOut(originalOut);
            Console.SetError(originalError);
            AnsiConsole.Console = originalAnsiConsole;
        }
    }

    /// <summary>
    /// The document is the last stdout line that parses as JSON. In <c>--json</c> mode narration is
    /// routed to stderr, so on a well-formed exit this is also the only line — but the contract is
    /// "stdout carries a document", not "stdout is exactly one line", and asserting the former is
    /// what a consumer relies on.
    /// </summary>
    private static JsonElement ParsePayload(string stdout)
    {
        var line = stdout.Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(l => l.Trim())
            .LastOrDefault(l => l.StartsWith('{'));
        Assert.False(line is null, "stdout carries no JSON document:\n" + stdout);
        return JsonDocument.Parse(line!).RootElement.Clone();
    }

    private static void WriteLinearCsv(string path, int rowCount, int seed, bool includeX2 = true)
    {
        var random = new Random(seed);
        var lines = new List<string> { includeX2 ? $"X1,X2,{LabelColumn}" : $"X1,{LabelColumn}" };
        for (int i = 0; i < rowCount; i++)
        {
            var x1 = random.NextDouble() * 10;
            var x2 = random.NextDouble() * 10;
            var y = 2 * x1 + 3 * x2 + 1;
            lines.Add(includeX2
                ? string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{x1:F4},{x2:F4},{y:F4}")
                : string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{x1:F4},{y:F4}"));
        }
        File.WriteAllLines(path, lines);
    }

    /// <summary>
    /// Trains a real regression model on X1,X2 → Y the way the product loads data (AutoML column
    /// inference folds the numerics into one Features vector), stores it as exp-001 with the input
    /// schema training would have recorded, and promotes it so <c>predict</c> resolves production.
    /// </summary>
    private async Task SeedRegressionWithSchemaAsync()
    {
        var trainCsv = Path.Combine(_testProjectRoot, "train.csv");
        WriteLinearCsv(trainCsv, rowCount: 50, seed: 42);

        var ml = new MLContext(seed: 42);
        var inference = ml.Auto().InferColumns(trainCsv, labelColumnName: LabelColumn, separatorChar: ',');
        var data = ml.Data.CreateTextLoader(inference.TextLoaderOptions).Load(trainCsv);
        var model = ml.Regression.Trainers.Sdca(labelColumnName: LabelColumn, featureColumnName: "Features").Fit(data);

        var experimentPath = _experimentStore.GetExperimentPath("default", "exp-001");
        Directory.CreateDirectory(experimentPath);
        ml.Model.Save(model, data.Schema, Path.Combine(experimentPath, ExperimentLayout.ModelFileName));

        await _experimentStore.SaveAsync("default", new ExperimentData
        {
            ModelName = "default",
            ExperimentId = "exp-001",
            Timestamp = DateTime.UtcNow,
            Status = "Completed",
            Task = "regression",
            Config = new ExperimentConfig
            {
                DataFile = trainCsv,
                LabelColumn = LabelColumn,
                TimeLimitSeconds = 10,
                Metric = "r_squared",
                TestSplit = 0.2,
                InputSchema = new InputSchemaInfo
                {
                    CapturedAt = DateTime.UtcNow,
                    Columns =
                    [
                        new ColumnSchema { Name = "X1", DataType = SchemaDataTypes.Numeric, Purpose = "Feature" },
                        new ColumnSchema { Name = "X2", DataType = SchemaDataTypes.Numeric, Purpose = "Feature" },
                        new ColumnSchema { Name = LabelColumn, DataType = SchemaDataTypes.Numeric, Purpose = "Label" },
                    ],
                },
            },
            Metrics = new Dictionary<string, double> { ["r_squared"] = 0.99 },
        }, CancellationToken.None);

        var (promoteExit, promoteOut, promoteErr) = await RunAsync("promote", "exp-001");
        Assert.True(promoteExit == 0, promoteOut + promoteErr);
    }

    [Fact]
    public async Task Predict_Json_SchemaMismatch_ExitsNonZeroWithAnErrorDocument()
    {
        await SeedRegressionWithSchemaAsync();
        var mismatched = Path.Combine(_testProjectRoot, "predict-missing-x2.csv");
        WriteLinearCsv(mismatched, rowCount: 5, seed: 7, includeX2: false);

        var (exitCode, stdout, stderr) = await RunAsync("predict", mismatched, "--json");

        Assert.Equal(1, exitCode);
        var payload = ParsePayload(stdout);
        var error = payload.GetProperty("error").GetString();
        Assert.NotNull(error);
        // The sentence only the schema branch emits — a "no model" or "file not found" exit would
        // also produce an error document, and this test exists for the branch behind the model load.
        Assert.StartsWith("Schema validation failed.", error);
        Assert.Contains("X2", error);
        // Nothing on stdout but the document: the four-part rendering went to stderr.
        Assert.DoesNotContain("Schema Validation Failed:", stdout);
        Assert.Contains("Schema Validation Failed:", stderr);
    }

    [Fact]
    public async Task Evaluate_Json_SchemaMismatch_ExitsNonZeroWithAnErrorDocument()
    {
        await SeedRegressionWithSchemaAsync();
        var mismatched = Path.Combine(_testProjectRoot, "test-missing-x2.csv");
        WriteLinearCsv(mismatched, rowCount: 5, seed: 7, includeX2: false);

        var (exitCode, stdout, stderr) = await RunAsync("evaluate", "exp-001", mismatched, "--json");

        Assert.Equal(1, exitCode);
        var payload = ParsePayload(stdout);
        var error = payload.GetProperty("error").GetString();
        Assert.NotNull(error);
        Assert.StartsWith("Schema validation failed.", error);
        Assert.Contains("X2", error);
        Assert.DoesNotContain("Schema Validation Failed:", stdout);
        Assert.Contains("Schema Validation Failed:", stderr);
    }

    /// <summary>
    /// The negative control: the same fixture, handed a file that matches, gets past the schema check
    /// and past the model — so the two tests above fail at the branch they name, not at an earlier
    /// exit a broken fixture would also reach.
    /// </summary>
    [Fact]
    public async Task Predict_Json_MatchingSchema_ReachesPredictions()
    {
        await SeedRegressionWithSchemaAsync();
        var matching = Path.Combine(_testProjectRoot, "predict-ok.csv");
        WriteLinearCsv(matching, rowCount: 5, seed: 7);

        var (exitCode, stdout, stderr) = await RunAsync("predict", matching, "--json");

        Assert.True(exitCode == 0, stdout + stderr);
        var payload = ParsePayload(stdout);
        Assert.False(payload.TryGetProperty("error", out _));
        Assert.Equal(5, payload.GetProperty("count").GetInt32());
    }
}
