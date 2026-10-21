using System.Globalization;
using System.Text.Json;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Models;
using MLoop.Core.Monitoring;
using MLoop.Core.Prediction;
using MLoop.Core.Storage;

namespace MLoop.Tests.ML;

/// <summary>
/// A real training run records the model's own answers beside its inputs, and records them the way a
/// prediction log does: scoring the training rows again through the serving path and comparing the
/// logged answers with the profile reads as no drift. A profile spelled differently from the log
/// (<c>True</c> where the log says <c>yes</c>) would put every answer in the "other" bin.
/// </summary>
[Collection("FileSystem")]
public class AnswerProfileTrainingTests : IDisposable
{
    private readonly string _tempDir;

    public AnswerProfileTrainingTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop-answer-profile-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(_tempDir, ".mloop"));
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    [Fact]
    public async Task A_classifier_profiles_its_answers_in_the_labels_it_was_trained_on()
    {
        var csv = Path.Combine(_tempDir, "churn.csv");
        var rnd = new Random(7);
        var lines = new List<string> { "tenure,charges,churned" };
        for (var i = 0; i < 300; i++)
        {
            var tenure = rnd.Next(1, 60);
            lines.Add($"{tenure},{rnd.Next(20, 120)},{(tenure < 20 ? "yes" : "no")}");
        }
        await File.WriteAllLinesAsync(csv, lines);

        var (profile, experimentPath, schema) = await TrainAsync(csv, "churned", "binary-classification");

        Assert.NotNull(profile.Prediction);
        Assert.False(profile.Prediction!.IsNumeric);
        Assert.Equal(DriftProfile.PredictionColumn, profile.Prediction.Column);
        Assert.Subset(new HashSet<string>(["yes", "no"], StringComparer.OrdinalIgnoreCase),
            profile.Prediction.Categories!.ToHashSet(StringComparer.OrdinalIgnoreCase));
        Assert.Equal(1.0, profile.Prediction.Shares.Sum(), 6);

        AssertTheLoggedAnswersDoNotDrift(profile, experimentPath, schema, csv, "churned", "binary-classification");
    }

    [Fact]
    public async Task A_regressor_profiles_its_predicted_values()
    {
        var csv = Path.Combine(_tempDir, "price.csv");
        var rnd = new Random(11);
        var lines = new List<string> { "size,rooms,price" };
        for (var i = 0; i < 300; i++)
        {
            var size = rnd.Next(30, 200);
            var rooms = rnd.Next(1, 6);
            lines.Add(string.Create(CultureInfo.InvariantCulture, $"{size},{rooms},{size * 3.5 + rooms * 10 + rnd.NextDouble() * 5:F2}"));
        }
        await File.WriteAllLinesAsync(csv, lines);

        var (profile, experimentPath, schema) = await TrainAsync(csv, "price", "regression");

        Assert.NotNull(profile.Prediction);
        Assert.True(profile.Prediction!.IsNumeric);
        Assert.NotEmpty(profile.Prediction.Edges!);

        AssertTheLoggedAnswersDoNotDrift(profile, experimentPath, schema, csv, "price", "regression");
    }

    private async Task<(DriftProfile Profile, string ExperimentPath, InputSchemaInfo Schema)> TrainAsync(string csv, string label, string task)
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
                ModelName = "answers",
                DataFile = csv,
                LabelColumn = label,
                Task = task,
                TimeLimitSeconds = 10,
            }, progress: null, CancellationToken.None);
        }
        finally
        {
            Console.SetOut(originalOut);
        }

        var experimentPath = store.GetExperimentPath("answers", result.ExperimentId);
        var json = await File.ReadAllTextAsync(Path.Combine(experimentPath, ExperimentLayout.DriftProfileFileName));
        var profile = JsonSerializer.Deserialize<DriftProfile>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;
        return (profile, experimentPath, result.Schema!);
    }

    /// <summary>Scores the training file through the serving path, logs each answer as serve does, and compares.</summary>
    private static void AssertTheLoggedAnswersDoNotDrift(
        DriftProfile profile, string experimentPath, InputSchemaInfo schema, string csv, string label, string task)
    {
        var rows = PredictionEngine.LoadCsvAsRows(csv, schema, label);
        var result = new PredictionEngine().PredictWithService(rows, schema, Path.Combine(experimentPath, ExperimentLayout.ModelFileName), task, label);

        // As the logs read them back: a JSON round trip of the logged output.
        var answers = result.Rows
            .Select(r => (object?)JsonDocument.Parse(JsonSerializer.Serialize(PredictionLogOutput.Of(r))).RootElement.Clone())
            .ToList();
        var inputs = rows.Select(r => (IReadOnlyDictionary<string, object?>)r.ToDictionary(kv => kv.Key, kv => (object?)kv.Value)).ToList();

        var report = DriftMeasure.Compare(profile, inputs, answers);

        Assert.NotNull(report.Prediction);
        Assert.True(report.Prediction!.Psi < DriftMeasure.ModerateThreshold, $"answer PSI {report.Prediction.Psi} ({report.Prediction.MostChangedBin})");
    }
}
