using System.Globalization;
using System.Text;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Models;
using MLoop.Core.Storage;

namespace MLoop.Tests.ML;

/// <summary>
/// A JSON row names its columns, and Hangul written as jamo (NFD) is the same name as Hangul written as syllables (NFC).
/// A value in another form was already matched as text; a key was matched as bytes, so a row from a client that decomposes
/// either failed with "none of the model's input columns" (every key in the other form) or - worse - scored with the
/// columns in the other form read as missing and an answer that showed nothing of it (some keys in each form).
/// </summary>
[Collection("FileSystem")]
public class ServedRowColumnNameFormSeamTests : IDisposable
{
    private const string Reading = "측정값";
    private const string Other = "기타";

    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), "mloop-served-key-form-tests", Guid.NewGuid().ToString("N"));

    public ServedRowColumnNameFormSeamTests() => Directory.CreateDirectory(Path.Combine(_tempDir, ".mloop"));

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    private static string Decomposed(string name) => name.Normalize(NormalizationForm.FormD);

    private static Dictionary<string, object>[] Row(string reading, string other) =>
        [new() { [reading] = 8.0, [other] = 2.0 }];

    [Fact]
    public async Task A_row_names_a_column_the_way_the_model_was_trained_or_in_the_other_Unicode_form()
    {
        var (modelPath, schema) = await TrainAsync(WriteTable());
        Assert.Contains(schema.Columns, c => c.Name == Reading);
        Assert.NotEqual(Reading, Decomposed(Reading)); // the two forms really are different strings
        var engine = new PredictionEngine();

        double Score(Dictionary<string, object>[] rows) =>
            engine.PredictWithService(rows, schema, modelPath, "regression", "y").Rows.Single().Score!.Value;

        var trained = Score(Row(Reading, Other));

        // every key decomposed, and only one of them
        Assert.Equal(trained, Score(Row(Decomposed(Reading), Decomposed(Other))), 3);
        Assert.Equal(trained, Score(Row(Decomposed(Reading), Other)), 3);
        Assert.Equal(trained, Score(Row(Reading, Decomposed(Other))), 3);

        // and the reading really moves the answer, or equality above would prove nothing
        Assert.True(Math.Abs(trained - Score(Row(Reading, Other).Select(r => { r[Reading] = 1.0; return r; }).ToArray())) > 5);
    }

    // y = 3·측정값 + 기타
    private string WriteTable()
    {
        var path = Path.Combine(_tempDir, "train.csv");
        var rnd = new Random(7);
        var lines = new List<string> { $"{Reading},{Other},y" };
        for (var i = 0; i < 300; i++)
        {
            var reading = rnd.NextDouble() * 10;
            var other = rnd.NextDouble() * 5;
            lines.Add(string.Create(CultureInfo.InvariantCulture, $"{reading:F3},{other:F3},{3 * reading + other + (rnd.NextDouble() - 0.5) * 0.3:F3}"));
        }
        File.WriteAllLines(path, lines);
        return path;
    }

    private async Task<(string ModelPath, InputSchemaInfo Schema)> TrainAsync(string csv)
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
                ModelName = "keyform",
                DataFile = csv,
                LabelColumn = "y",
                Task = "regression",
                TimeLimitSeconds = 60,
                MaxTrials = 10,
            }, progress: null, CancellationToken.None);
        }
        finally
        {
            Console.SetOut(originalOut);
        }

        return (Path.Combine(store.GetExperimentPath("keyform", result.ExperimentId), ExperimentLayout.ModelFileName), result.Schema!);
    }
}
