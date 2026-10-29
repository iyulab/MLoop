using System.Globalization;
using System.Text;
using System.Text.Json;
using MLoop.Core.Embedding;
using MLoop.Core.Models;
using MLoop.Core.Storage;

namespace MLoop.Tests.Commands;

/// <summary>
/// A text column declared with <c>encoder:</c> in mloop.yaml is trained on as its sentence vectors, end to end through the
/// command tree: the experiment records the encoder beside its model, predict and evaluate read the column's text
/// through the same encoder, and a model whose encoder is gone or replaced says so instead of scoring.
/// </summary>
[Collection("FileSystem")]
public sealed class TextEncoderTrainingTests : IDisposable
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures", "embedding");
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string? _previousRoot = Environment.GetEnvironmentVariable(EncoderStore.RootVariable);
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-textenc-" + Guid.NewGuid().ToString("N"));

    private static readonly string[] Positive = ["배송이 빨라서 좋았어요", "상품이 좋아요 다시 살게요", "포장이 꼼꼼해요", "가격 대비 만족합니다"];
    private static readonly string[] Negative = ["배송이 늦어서 별로였어요", "색상이 사진과 달라요", "the product is bad", "colour differs from the photo"];
    private static readonly string[] Neutral = ["그냥 보통이에요", "무난한 편입니다", "it is okay", "average product"];

    private static readonly EncoderDefinition TinyText = new(
        "tiny-text", "1", 4, null, "input_ids", "last_hidden_state", EncoderPooling.MeanOfTokens,
        Kind: EncoderKind.Text, Text: new TextPreprocessing(12));

    public TextEncoderTrainingTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
        Environment.SetEnvironmentVariable(EncoderStore.RootVariable, Path.Combine(_workspace, "encoders"));
        new EncoderStore().Add(TinyText, Path.Combine(Fixtures, "tiny-text-encoder.onnx"), Path.Combine(Fixtures, "tiny-tokenizer.model"));
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(EncoderStore.RootVariable, _previousRoot);
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private static string Reviews(int rows, int seed, bool withLabel = true, bool threeClasses = false)
    {
        var csv = new StringBuilder(withLabel ? "review,x,y\n" : "review,x\n");
        var rng = new Random(seed);
        for (var i = 0; i < rows; i++)
        {
            var pick = threeClasses ? rng.Next(3) : (rng.NextDouble() < 0.5 ? 0 : 1);
            var (pool, label) = pick switch { 0 => (Positive, "yes"), 1 => (Negative, "no"), _ => (Neutral, "maybe") };
            var text = pool[rng.Next(pool.Length)];
            csv.Append(CultureInfo.InvariantCulture, $"{text},{rng.NextDouble():F4}");
            csv.Append(withLabel ? "," + label + "\n" : "\n");
        }
        return csv.ToString();
    }

    private async Task<string> TrainAsync(string task = "binary-classification", bool threeClasses = false)
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", task)).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("mloop.yaml", $"""
            project: p
            models:
              default:
                task: {task}
                label: y
                columns:
                  review:
                    type: text
                    encoder: tiny-text
            """);
        await File.WriteAllTextAsync("datasets/train.csv", Reviews(threeClasses ? 360 : 240, 3, threeClasses: threeClasses));
        await File.WriteAllTextAsync("datasets/predict.csv", Reviews(20, 5, withLabel: false));
        var (exit, output, error) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "60", "--max-trials", "10");
        Assert.True(exit == 0, output + Environment.NewLine + error);
        return Path.Combine("models", "default", "staging", "exp-001", ExperimentLayout.ModelFileName);
    }

    [Fact]
    public async Task ADeclaredEncoder_IsTrainedOn_RecordedBesideTheModel_AndReadAgainByPredictAndEvaluate()
    {
        var modelPath = await TrainAsync();

        var encodings = TextEncodings.ReadBeside(modelPath);
        var column = Assert.Single(encodings!.Columns);
        Assert.Equal("review", column.Column);
        Assert.Equal(("tiny-text", EncoderKind.Text), (column.Encoder.Id, column.Encoder.Kind));
        Assert.NotNull(column.Encoder.TokenizerSha256);
        Assert.True(File.Exists(Path.Combine("models", "default", "production", ExperimentLayout.TextEncodersFileName)));
        var report = File.ReadAllText(Path.Combine(Path.GetDirectoryName(modelPath)!, ExperimentLayout.ReportFileName));
        Assert.Contains("| Text encoder | `review` → tiny-text 1 (4 dimensions", report);
        Assert.DoesNotContain("textEncodings", File.ReadAllText(Path.Combine(Path.GetDirectoryName(modelPath)!, ExperimentLayout.MetadataFileName)));

        var (predicted, json, predictErr) = await CliRunner.RunAsync("predict", "datasets/predict.csv", "--json");
        Assert.True(predicted == 0, json + predictErr);
        Assert.Equal(20, JsonDocument.Parse(json).RootElement.GetProperty("predictions").GetArrayLength());

        // The texts decide the label, and the model reads them only through the encoder.
        await File.WriteAllTextAsync("datasets/test.csv", Reviews(80, 11));
        var (evaluated, metrics, evaluateErr) = await CliRunner.RunAsync("evaluate", "exp-001", "datasets/test.csv", "--json");
        Assert.True(evaluated == 0, metrics + evaluateErr);
        Assert.True(JsonDocument.Parse(metrics).RootElement.GetProperty("testMetrics").GetProperty("accuracy").GetDouble() > 0.9, metrics);
    }

    /// <summary>
    /// The threshold and calibration commands load the file for the model through the same reader as predict and
    /// evaluate — a text column read through an encoder has to come out as the same sentence vectors there, or they
    /// stop at "Could not find input column" on a model that trained, promoted and evaluated without a word.
    /// </summary>
    [Fact]
    public async Task TheThresholdAndCalibrationRows_ReadAnEncodedTextColumnToo()
    {
        await TrainAsync();
        await File.WriteAllTextAsync("datasets/test.csv", Reviews(120, 11));

        var (thresholded, thresholdJson, thresholdErr) = await CliRunner.RunAsync(
            "threshold", "exp-001", "--precision", "0.9", "--data", "datasets/test.csv", "--json");
        Assert.True(thresholded == 0, thresholdJson + thresholdErr);
        Assert.True(JsonDocument.Parse(thresholdJson).RootElement.TryGetProperty("threshold", out _), thresholdJson);

        // A threshold stands on the model, and calibration is refused while one stands: clear it first.
        Assert.Equal(0, (await CliRunner.RunAsync("threshold", "exp-001", "--clear")).ExitCode);
        var (calibrated, calibrationJson, calibrationErr) = await CliRunner.RunAsync(
            "calibrate", "exp-001", "--data", "datasets/test.csv", "--json");
        Assert.True(calibrated == 0, calibrationJson + calibrationErr);
        Assert.True(File.Exists(Path.Combine("models", "default", "staging", "exp-001", ExperimentLayout.CalibrationFileName)));
    }

    [Fact]
    public async Task TheCalibrationRow_ReadsAnEncodedTextColumnOfAMulticlassModelToo()
    {
        await TrainAsync("multiclass-classification", threeClasses: true);
        await File.WriteAllTextAsync("datasets/test.csv", Reviews(150, 11, threeClasses: true));

        var (calibrated, calibrationJson, calibrationErr) = await CliRunner.RunAsync(
            "calibrate", "exp-001", "--data", "datasets/test.csv", "--json");

        Assert.True(calibrated == 0, calibrationJson + calibrationErr);
        Assert.True(File.Exists(Path.Combine("models", "default", "staging", "exp-001", ExperimentLayout.CalibrationFileName)));
    }

    [Fact]
    public async Task AModelWhoseEncoderIsGoneOrReplaced_RefusesToPredict_NamingTheEncoder()
    {
        await TrainAsync();

        // Same id, other weights: the vectors would be another space.
        var other = Path.Combine(_workspace, "other.onnx");
        File.Copy(Path.Combine(Fixtures, "tiny-text-encoder-no-types.onnx"), other);
        new EncoderStore().Add(TinyText, other, Path.Combine(Fixtures, "tiny-tokenizer.model"));
        var (replaced, _, replacedErr) = await CliRunner.RunAsync("predict", "datasets/predict.csv");
        Assert.NotEqual(0, replaced);
        Assert.Contains("tiny-text", replacedErr);
        Assert.Contains("weights", replacedErr);

        new EncoderStore().Remove("tiny-text");
        var (gone, _, goneErr) = await CliRunner.RunAsync("predict", "datasets/predict.csv");
        Assert.NotEqual(0, gone);
        Assert.Contains("not installed", goneErr);
    }

    [Fact]
    public async Task AnEncoderOnAColumnThatIsNotText_OrForATaskWithoutFeatureColumns_IsAValidationError()
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "clustering")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("mloop.yaml", """
            project: p
            models:
              default:
                task: clustering
                columns:
                  review:
                    type: categorical
                    encoder: tiny-text
            """);

        var (exit, stdout, stderr) = await CliRunner.RunAsync("validate");

        Assert.NotEqual(0, exit);
        Assert.Contains("set type: text", stdout + stderr);
        Assert.Contains("does not read them", stdout + stderr);
    }
}
