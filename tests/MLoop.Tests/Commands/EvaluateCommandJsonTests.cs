using System.CommandLine;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLoop.CLI;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Models;
using MLoop.Core.Storage;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>evaluate --json</c> has had no automated coverage since it shipped — every existing test in
/// this file's sibling (<c>EvaluateCommandTests</c>) is a string-level unit test of a pure helper,
/// never exercised through the real command tree. Unlike <c>status</c>/<c>runtime list</c>/
/// <c>prep run</c>, <c>evaluate</c> actually loads and scores a trained ML.NET model against test
/// data, so the fixture here trains a tiny real regression model (no AutoML search — a direct SDCA
/// fit is fast and deterministic) rather than faking metadata alone.
/// </summary>
[Collection("FileSystem")]
public class EvaluateCommandJsonTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly string _originalDirectory;
    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;
    private readonly IExperimentStore _experimentStore;

    public EvaluateCommandJsonTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();

        _testProjectRoot = Path.Combine(Path.GetTempPath(), "mloop-evaluate-json-test-" + Guid.NewGuid());
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

    private const string LabelColumn = "Y";

    /// <summary>Deterministic Y = 2*X1 + 3*X2 + 1, so a fitted model scores near-perfectly.</summary>
    private static void WriteLinearCsv(string path, int rowCount, int seed)
    {
        var random = new Random(seed);
        var lines = new List<string> { $"X1,X2,{LabelColumn}" };
        for (int i = 0; i < rowCount; i++)
        {
            var x1 = random.NextDouble() * 10;
            var x2 = random.NextDouble() * 10;
            var y = 2 * x1 + 3 * x2 + 1;
            lines.Add(string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{x1:F4},{x2:F4},{y:F4}"));
        }
        File.WriteAllLines(path, lines);
    }

    /// <summary>
    /// Trains a tiny real SDCA regression model (deterministic linear relationship, no AutoML
    /// search) and writes it as the experiment's model.zip, alongside metadata via
    /// <see cref="ExperimentStore"/> so <c>ExperimentExists</c>/<c>LoadAsync</c> resolve it exactly
    /// as a real trained experiment would. Returns the test CSV path to evaluate against.
    /// </summary>
    private async Task<string> CreateRegressionFixtureAsync(string modelName, string experimentId)
    {
        // Train from a CSV loaded exactly the way the product loads it: AutoML column inference
        // folds the non-label numerics into a single "Features" vector, and the fitted pipeline
        // consumes that vector. Training from a POCO with individually-named columns instead would
        // produce a model no MLoop command could feed, so the fixture would prove nothing.
        var trainCsvPath = Path.Combine(_testProjectRoot, "train.csv");
        WriteLinearCsv(trainCsvPath, rowCount: 50, seed: 42);

        var mlContext = new MLContext(seed: 42);
        var inference = mlContext.Auto().InferColumns(trainCsvPath, labelColumnName: LabelColumn, separatorChar: ',');
        var trainData = mlContext.Data.CreateTextLoader(inference.TextLoaderOptions).Load(trainCsvPath);

        var model = mlContext.Regression.Trainers
            .Sdca(labelColumnName: LabelColumn, featureColumnName: "Features")
            .Fit(trainData);

        var experimentPath = _experimentStore.GetExperimentPath(modelName, experimentId);
        Directory.CreateDirectory(experimentPath);
        var modelPath = Path.Combine(experimentPath, ExperimentLayout.ModelFileName);
        // Training saves with a null schema, so the fixture does too — a schema-carrying
        // save would put the model on a code path no MLoop-produced model takes.
        mlContext.Model.Save(model, null, modelPath);

        var experiment = new ExperimentData
        {
            ModelName = modelName,
            ExperimentId = experimentId,
            Timestamp = DateTime.UtcNow,
            Status = "Completed",
            Task = "regression",
            Config = new ExperimentConfig
            {
                DataFile = trainCsvPath,
                LabelColumn = LabelColumn,
                TimeLimitSeconds = 10,
                Metric = "r_squared",
                TestSplit = 0.2,
                // Real training always captures this; without it every consumer falls back to
                // AutoML column inference, which is not the path a trained experiment takes.
                InputSchema = new InputSchemaInfo
                {
                    CapturedAt = DateTime.UtcNow,
                    Columns =
                    [
                        new ColumnSchema { Name = "X1", DataType = "Single", Purpose = "Feature" },
                        new ColumnSchema { Name = "X2", DataType = "Single", Purpose = "Feature" },
                        new ColumnSchema { Name = LabelColumn, DataType = "Single", Purpose = "Label" }
                    ]
                }
            },
            Metrics = new Dictionary<string, double> { ["r_squared"] = 0.95 }
        };
        await _experimentStore.SaveAsync(modelName, experiment, CancellationToken.None);

        var testCsvPath = Path.Combine(_testProjectRoot, "test.csv");
        WriteLinearCsv(testCsvPath, rowCount: 10, seed: 7);

        return testCsvPath;
    }

    // Same AnsiConsole.Console rebinding as the other --json tests. Stderr is captured too, so a
    // failing run reports the actual cause (ErrorConsole writes there) instead of an bare exit code.
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
            AnsiConsole.Console = AnsiConsole.Create(new AnsiConsoleSettings
            {
                Out = new AnsiConsoleOutput(buffer)
            });

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

    [Fact]
    public async Task Evaluate_Json_RealModel_ReportsTestMetrics()
    {
        var testCsvPath = await CreateRegressionFixtureAsync("default", "exp-001");

        var (exitCode, stdout, stderr) = await RunAsync("evaluate", "exp-001", testCsvPath, "--json");

        Assert.True(exitCode == 0, stdout + stderr);
        using var doc = System.Text.Json.JsonDocument.Parse(stdout);
        Assert.Equal("default", doc.RootElement.GetProperty("model").GetString());
        Assert.Equal("exp-001", doc.RootElement.GetProperty("experimentId").GetString());
        var testMetrics = doc.RootElement.GetProperty("testMetrics");
        Assert.True(testMetrics.GetProperty("r_squared").GetDouble() > 0.9); // near-perfect linear fit
        Assert.True(doc.RootElement.GetProperty("trainingMetrics").GetProperty("r_squared").GetDouble() > 0);
    }

    [Fact]
    public async Task Evaluate_Json_UnknownExperiment_EmitsErrorPayload()
    {
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, "models", "default", "staging"));

        var (exitCode, stdout, stderr) = await RunAsync("evaluate", "exp-999", "nope.csv", "--json");

        Assert.Equal(1, exitCode);

        // This assertion used to read `Assert.Equal("", stdout)`, on the reasoning that an error exit
        // has nothing valid to say and so keeps stdout pure. That was the defect, not the contract: an
        // empty stdout fails JSON.parse exactly as prose does, and a consumer cannot tell it apart from
        // a crash. An error exit owes the same thing every other exit owes — a document.
        var doc = System.Text.Json.JsonDocument.Parse(stdout);
        Assert.False(string.IsNullOrWhiteSpace(doc.RootElement.GetProperty("error").GetString()));
        Assert.NotEqual("", stderr);
    }

    [Fact]
    public async Task Evaluate_Human_Unaffected_ByJsonChanges()
    {
        var testCsvPath = await CreateRegressionFixtureAsync("default", "exp-001");

        var (exitCode, stdout, stderr) = await RunAsync("evaluate", "exp-001", testCsvPath);

        Assert.True(exitCode == 0, stdout + stderr);
        Assert.Contains("Evaluation Results", stdout);
        Assert.DoesNotContain("{", stdout); // no JSON leaked into the human report
    }
}
