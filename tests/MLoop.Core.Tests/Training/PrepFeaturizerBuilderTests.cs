using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Preprocessing;
using MLoop.Core.Training;

namespace MLoop.Core.Tests.Training;

public class PrepFeaturizerBuilderTests
{
    [Fact]
    public void Build_returns_null_when_no_statistical_steps()
    {
        var ctx = new MLContext(seed: 42);
        var steps = new List<PrepStep> { new() { Type = "remove-columns", Columns = new() { "id" } } };
        Assert.Null(new PrepFeaturizerBuilder().Build(ctx, steps));
    }

    [Fact]
    public void Build_creates_estimator_for_normalize_minmax()
    {
        var ctx = new MLContext(seed: 42);
        var steps = new List<PrepStep>
        {
            new() { Type = "normalize", Method = "min-max", Columns = new() { "age" } }
        };
        var est = new PrepFeaturizerBuilder().Build(ctx, steps);
        Assert.NotNull(est);
    }

    private sealed class QuietLogger : MLoop.Extensibility.Preprocessing.ILogger
    {
        public void Info(string message) { }
        public void Warning(string message) { }
        public void Error(string message) { }
        public void Error(string message, Exception exception) { }
        public void Debug(string message) { }
    }

    private sealed class AgeRow { public float Age { get; set; } }

    // What a normalize step does to the column [1, 2, 3, 4, 5], fitted and applied.
    private static float[] NormalizedAge(string? method)
    {
        var ctx = new MLContext(seed: 42);
        var data = ctx.Data.LoadFromEnumerable(Enumerable.Range(1, 5).Select(i => new AgeRow { Age = i }));
        var steps = new List<PrepStep> { new() { Type = "normalize", Method = method, Columns = new() { "Age" } } };
        var estimator = new PrepFeaturizerBuilder().Build(ctx, steps)!;
        return estimator.Fit(data).Transform(data).GetColumn<float>("Age").ToArray();
    }

    private static async Task<float[]> ExecutorNormalizedAge(string method)
    {
        var dir = Path.Combine(Path.GetTempPath(), "norm-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var input = Path.Combine(dir, "a.csv");
            var output = Path.Combine(dir, "out.csv");
            await File.WriteAllLinesAsync(input, ["Age", "1", "2", "3", "4", "5"]);
            await new DataPipelineExecutor(new QuietLogger()).ExecuteAsync(
                input, [new PrepStep { Type = "normalize", Method = method, Columns = new() { "Age" } }], output);
            return [.. File.ReadAllLines(output).Skip(1).Where(l => l.Trim().Length > 0)
                .Select(l => float.Parse(l.Trim(), System.Globalization.CultureInfo.InvariantCulture))];
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Theory]
    [InlineData("min-max")]
    [InlineData("z-score")]
    public async Task ANormalizeStepFittedInsideTraining_GivesTheValuesTheExecutorGivesTheSameStepOnTheFile(string method)
    {
        // The step is the same whichever stage runs it (a task that cannot fit it inside training runs it on the file), so
        // the two have to agree on what the name means.
        var inTraining = NormalizedAge(method);
        var onTheFile = await ExecutorNormalizedAge(method);

        Assert.Equal(onTheFile.Length, inTraining.Length);
        for (var i = 0; i < onTheFile.Length; i++)
            Assert.Equal(onTheFile[i], inTraining[i], 3);
    }

    [Fact]
    public void MinMax_PutsTheSmallestValueAtZeroAndTheLargestAtOne_NotTheValueOverTheLargest()
    {
        var values = NormalizedAge("min-max");

        Assert.Equal(0f, values.Min(), 5);
        Assert.Equal(1f, values.Max(), 5);
        Assert.Equal(0.5f, values[2], 5);   // the middle of 1..5
    }

    [Fact]
    public void ZScore_CentersTheColumnOnItsMean()
    {
        var values = NormalizedAge("z-score");

        Assert.Equal(0f, values.Average(), 4);
        Assert.Equal(0f, values[2], 4);   // 3 is the mean of 1..5
    }

    [Theory]
    [InlineData("zscore")]
    [InlineData("z_score")]
    [InlineData("standard")]
    [InlineData("Z-SCORE")]
    public void EverySpellingTheExecutorReadsAsZScore_IsZScoreInsideTraining(string spelling)
        => Assert.Equal(NormalizedAge("z-score"), NormalizedAge(spelling));

    [Theory]
    [InlineData("minmax")]
    [InlineData("min_max")]
    [InlineData(null)]
    public void EverySpellingTheExecutorReadsAsMinMax_AndNoMethodAtAll_IsMinMaxInsideTraining(string? spelling)
        => Assert.Equal(NormalizedAge("min-max"), NormalizedAge(spelling));

    [Theory]
    [InlineData("foo")]
    [InlineData("standardise")]
    public void AMethodNoSpellingNames_IsRefusedInsteadOfFittedAsMinMax(string method)
    {
        var error = Assert.Throws<InvalidOperationException>(() => NormalizedAge(method));

        Assert.Contains($"Unknown normalization method: '{method}'", error.Message);
        Assert.Contains("min-max, z-score, robust", error.Message);
    }

    [Theory]
    [InlineData("normalize", "foo", "'normalize' method 'foo' is not one of: min-max, z-score, robust")]
    [InlineData("scale", "zscorre", "'scale' method 'zscorre' is not one of: min-max, z-score, robust")]
    [InlineData("fill-missing", "meen", "'fill-missing' method 'meen' is not one of: mean, median, mode, forward, backward, constant")]
    public void TheValidatorsSayWhichMethodsAStepTakes(string type, string method, string expected)
    {
        var step = new PrepStep { Type = type, Method = method, Columns = new() { "Age" } };

        Assert.Contains(expected, PrepStepRules.Problems(step));
    }

    [Theory]
    [InlineData("normalize", "zscore")]
    [InlineData("normalize", "standard")]
    [InlineData("normalize", null)]
    [InlineData("fill-missing", "average")]
    [InlineData("fill-missing", "ffill")]
    [InlineData("fill-missing", "bfill")]
    [InlineData("fill-missing", null)]
    public void ASpellingTheExecutorReads_IsNoProblem(string type, string? method)
        => Assert.Empty(PrepStepRules.Problems(new PrepStep { Type = type, Method = method, Columns = new() { "Age" } }));

    [Theory]
    [InlineData("mean", PrepCategory.PreFeaturizer)]
    [InlineData("average", PrepCategory.PreFeaturizer)]   // the mean, as the executor reads it: fitted on the training rows, not the whole file
    [InlineData(null, PrepCategory.PreFeaturizer)]        // no method is the mean
    [InlineData("median", PrepCategory.UnsupportedLeakageWarn)]
    [InlineData("constant", PrepCategory.CsvStage)]
    [InlineData("ffill", PrepCategory.CsvStage)]
    [InlineData("meen", PrepCategory.CsvStage)]           // unknown: stays where the executor refuses it
    public void AFillMissingStepIsRoutedByWhatItsMethodMeans(string? method, PrepCategory expected)
        => Assert.Equal(expected, PrepStepClassifier.Classify(new PrepStep { Type = "fill-missing", Method = method, Columns = new() { "Age" } }));
}
