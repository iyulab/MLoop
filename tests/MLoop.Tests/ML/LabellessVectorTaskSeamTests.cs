using System.Globalization;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.Core.Storage;

namespace MLoop.Tests.ML;

/// <summary>
/// Clustering and anomaly detection without a label read one numeric vector, and every reader must build
/// the same one: the training run (which also profiles the model's answers), <c>mloop predict</c> on a
/// file, and the serving path on rows. Each step was tested on its own while the seams between them were
/// not — so a label-less clustering run failed at its own answer profile and a label-less anomaly model
/// could not predict, on plain numeric data, with every suite green. These tests go through all three in
/// one fixture.
/// </summary>
[Collection("FileSystem")]
public class LabellessVectorTaskSeamTests : IDisposable
{
    private readonly string _tempDir;

    private readonly string _originalDirectory = Directory.GetCurrentDirectory();

    public LabellessVectorTaskSeamTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop-labelless-seam-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(_tempDir, ".mloop"));
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    // The commands a user runs, in a project — `mloop train` does more than the engine alone (its own
    // run of the answer profile among them), and a label-less clustering run failed only there.
    [Theory]
    [InlineData("clustering", false)]
    [InlineData("clustering", true)]
    [InlineData("anomaly-detection", false)]
    [InlineData("anomaly-detection", true)]
    public async Task A_label_less_project_trains_and_predicts_through_the_commands(string task, bool withYesNoColumn)
    {
        var workspace = Path.Combine(_tempDir, "cli");
        Directory.CreateDirectory(workspace);
        Directory.SetCurrentDirectory(workspace);
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", task)).ExitCode);
        var project = Path.Combine(workspace, "p");
        Directory.SetCurrentDirectory(project);
        File.Copy(WriteTable("train.csv", 300, withYesNoColumn), Path.Combine(project, "datasets", "train.csv"), true);
        File.Copy(WriteTable("predict.csv", 8, withYesNoColumn), Path.Combine(project, "datasets", "p.csv"), true);

        var train = await CliRunner.RunAsync("train", "datasets/train.csv", "--time", "10");   // no search: clustering and anomaly detection fit one pipeline
        Assert.True(train.ExitCode == 0, train.Stdout + train.Stderr);

        var predict = await CliRunner.RunAsync("predict", "datasets/p.csv", "--json");
        Assert.True(predict.ExitCode == 0, predict.Stdout + predict.Stderr);
        Assert.Contains("\"count\":8", predict.Stdout);
    }

    [Theory]
    [InlineData("clustering", false)]
    [InlineData("clustering", true)]
    [InlineData("anomaly-detection", false)]
    [InlineData("anomaly-detection", true)]
    public async Task A_label_less_model_trains_and_predicts_through_every_reader(string task, bool withYesNoColumn)
    {
        var csv = WriteTable("train.csv", 200, withYesNoColumn);
        var predictCsv = WriteTable("predict.csv", 8, withYesNoColumn);

        var (experimentPath, schema) = await TrainAsync(csv, task);
        var modelPath = Path.Combine(experimentPath, ExperimentLayout.ModelFileName);
        Assert.True(File.Exists(Path.Combine(experimentPath, ExperimentLayout.DriftProfileFileName)));

        // mloop predict on a file
        var output = Path.Combine(_tempDir, $"{task}-{withYesNoColumn}-out.csv");
        var count = await new PredictionEngine().PredictAsync(
            modelPath, predictCsv, output, schema,
            CategoricalMapper.UnknownValueStrategy.Auto,
            cancellationToken: default,
            labelColumnOverride: null,
            preserveColumns: null,
            interval: null,
            taskType: task);
        Assert.Equal(8, count);

        // The serving path, on rows
        var rows = PredictionEngine.LoadCsvAsRows(predictCsv, schema, null);
        var served = new PredictionEngine().PredictWithService(rows, schema, modelPath, task, null);
        Assert.Equal(8, served.Rows.Count);
    }

    // x first: the column that used to stand in as a label and be set apart from the vector.
    private string WriteTable(string name, int rows, bool withYesNoColumn)
    {
        var path = Path.Combine(_tempDir, name);
        var rnd = new Random(name.Length);
        var lines = new List<string> { withYesNoColumn ? "x,flag,y,z" : "x,y,z" };
        for (var i = 0; i < rows; i++)
        {
            var center = i % 2 == 0 ? 2.0 : 8.0;
            var x = (center + rnd.NextDouble()).ToString("F3", CultureInfo.InvariantCulture);
            var y = (center * 3 + rnd.NextDouble()).ToString("F3", CultureInfo.InvariantCulture);
            var z = (i % 7).ToString(CultureInfo.InvariantCulture);
            lines.Add(withYesNoColumn ? $"{x},{(i % 3 == 0 ? "yes" : "no")},{y},{z}" : $"{x},{y},{z}");
        }
        File.WriteAllLines(path, lines);
        return path;
    }

    private async Task<(string ExperimentPath, InputSchemaInfo Schema)> TrainAsync(string csv, string task)
    {
        var fs = new FileSystemManager();
        var store = new ExperimentStore(fs, new ProjectDiscovery(fs), _tempDir);
        var originalOut = Console.Out;
        Console.SetOut(new StringWriter());
        TrainingResult result;
        try
        {
            result = await new TrainingEngine(fs, store).TrainAsync(new TrainingConfig
            {
                ModelName = "seam",
                DataFile = csv,
                LabelColumn = "",
                Task = task,   // no search: clustering and anomaly detection fit one pipeline
                TimeLimitSeconds = 10,
                NumClusters = 0,
            }, progress: null, CancellationToken.None);
        }
        finally
        {
            Console.SetOut(originalOut);
        }

        return (store.GetExperimentPath("seam", result.ExperimentId), result.Schema!);
    }
}
