using Microsoft.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Training;
using MLoop.Tests.Common;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// A regression label the loader read as text used to fail every trainer with the same opaque
/// "Schema mismatch for label column: expected Single, got String", four times over. It is
/// refused up front now, saying which values did it — and only when the loader really read the
/// label as text: N/A, NA, NaN and ? are missing numbers to the loader, and those runs train.
/// </summary>
[Collection("FileSystem")]
[Trait(TestCategories.Category, TestCategories.Integration)]
public class NumericLabelCheckTests : IDisposable
{
    private readonly string _tmpDir;
    private readonly MLContext _ctx = new(seed: 42);

    public NumericLabelCheckTests()
    {
        _tmpDir = Path.Combine(Path.GetTempPath(), $"mloop_numlabel_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tmpDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tmpDir))
            Directory.Delete(_tmpDir, recursive: true);
    }

    /// <summary>Rows of <c>a,b,y</c> where <paramref name="odd"/> replaces the label of rows 10, 50 and 90.</summary>
    private string Write(string name, Func<double, string>? odd)
    {
        var lines = new List<string> { "a,b,y" };
        for (var i = 0; i < 300; i++)
        {
            var a = (i * 7) % 100 / 10.0;
            var b = (i * 13) % 50 / 10.0;
            var y = Math.Round(2 * a + b + ((i * 37) % 11 - 5) * 0.05, 3);
            var label = odd is not null && i is 10 or 50 or 90 ? odd(y) : y.ToString(System.Globalization.CultureInfo.InvariantCulture);
            lines.Add($"{a.ToString(System.Globalization.CultureInfo.InvariantCulture)},{b.ToString(System.Globalization.CultureInfo.InvariantCulture)},{label}");
        }

        var path = Path.Combine(_tmpDir, name);
        File.WriteAllLines(path, lines);
        return path;
    }

    private string? RefusalFor(string path, string task = "regression")
        => NumericLabelCheck.Refusal(new CsvDataLoader(_ctx).LoadData(path, "y", task), task, "y");

    [Fact]
    public void ANumberWithAThousandsSeparator_MakesTheLabelText_AndIsNamedWithItsFix()
    {
        var refusal = RefusalFor(Write("comma.csv", _ => "\"1,234.5\""));

        Assert.NotNull(refusal);
        Assert.Contains("Label column 'y' is read as text", refusal, StringComparison.Ordinal);
        Assert.Contains("'1,234.5'", refusal, StringComparison.Ordinal);
        Assert.Contains("written for people", refusal, StringComparison.Ordinal);
        Assert.Contains("plain numbers", refusal, StringComparison.Ordinal);
    }

    [Fact]
    public void APercentSign_IsTheSameKindOfRefusal()
    {
        var refusal = RefusalFor(Write("percent.csv", y => $"{Math.Round(y)}%"));

        Assert.NotNull(refusal);
        Assert.Contains("written for people", refusal, StringComparison.Ordinal);
    }

    [Fact]
    public void ADecimalComma_IsNamedAsSuch_NotSentToClassification()
    {
        var refusal = RefusalFor(Write("decimal.csv", _ => "\"1,5\""));

        Assert.NotNull(refusal);
        Assert.Contains("decimal comma", refusal, StringComparison.Ordinal);
        Assert.Contains("'1,5'", refusal, StringComparison.Ordinal);
        Assert.Contains("1.5 for 1,5", refusal, StringComparison.Ordinal);
        Assert.DoesNotContain("multiclass-classification", refusal, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("recommendation")]
    [InlineData("ranking")]
    [InlineData("forecasting")]
    public void EveryTaskWhoseLabelIsANumber_IsAskedAboutIt(string task)
    {
        Assert.True(NumericLabelCheck.HasNumericLabel(task));
        Assert.NotNull(RefusalFor(Write($"{task}.csv", _ => "\"1,234.5\""), task));
    }

    [Theory]
    [InlineData("binary-classification")]
    [InlineData("multiclass-classification")]
    [InlineData("text-classification")]
    [InlineData("ner")]
    [InlineData("question-answering")]
    [InlineData("clustering")]
    [InlineData("anomaly-detection")]
    [InlineData("image-classification")]
    [InlineData("object-detection")]
    public void ATaskWhoseLabelIsAClassOrTextOrNone_IsNotAskedAboutIt(string task)
        => Assert.False(NumericLabelCheck.HasNumericLabel(task));

    [Fact]
    public void AWordAmongTheNumbers_IsRefusedAndPointsToClassification()
    {
        var refusal = RefusalFor(Write("word.csv", _ => "unknown"));

        Assert.NotNull(refusal);
        Assert.Contains("'unknown'", refusal, StringComparison.Ordinal);
        Assert.Contains("multiclass-classification", refusal, StringComparison.Ordinal);
        Assert.DoesNotContain("written for people", refusal, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("N/A")]
    [InlineData("NA")]
    [InlineData("NaN")]
    [InlineData("?")]
    [InlineData("")]
    public void AMissingNumber_IsNotRefused_BecauseTheLoaderKeepsTheLabelNumeric(string missing)
    {
        Assert.Null(RefusalFor(Write("missing.csv", _ => missing)));
    }

    [Fact]
    public void PlainNumbers_AreNotRefused()
    {
        Assert.Null(RefusalFor(Write("plain.csv", odd: null)));
    }

    [Fact]
    public void ATaskThatDoesNotNeedANumber_IsNeverAskedAboutTheLabel()
    {
        // A class label is text by nature.
        Assert.Null(RefusalFor(Write("classes.csv", _ => "unknown"), task: "multiclass-classification"));
    }

    [Fact]
    public async Task TheRunStopsBeforeTheSearch_WithTheRefusal()
    {
        var path = Write("run.csv", _ => "\"1,234.5\"");
        var runner = new TrainingRunner(_ctx, new CsvDataLoader(_ctx));

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => runner.RunAsync(new TrainingConfig
        {
            ModelName = "numlabel",
            DataFile = path,
            LabelColumn = "y",
            Task = "regression",
            TimeLimitSeconds = 20,
            MaxTrials = 12,
            Metric = string.Empty
        }, progress: null, CancellationToken.None));

        Assert.Contains("Label column 'y' is read as text", ex.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("Schema mismatch", ex.Message, StringComparison.Ordinal);
    }

    // The same question asked of a file about to be scored (evaluate, explain, threshold, calibrate): there is no run that
    // typed the column from its first lines, so the values are counted over every row of the file itself.
    [Fact]
    public void AFileAboutToBeScored_IsRefused_WithTheCountOverEveryRow()
    {
        var path = Write("scored.csv", _ => "\"1,234.5\"");

        var refusal = NumericLabelCheck.FileRefusal(path, "y", "regression");

        Assert.NotNull(refusal);
        Assert.Contains("3 of the 300 label values in the file", refusal, StringComparison.Ordinal);
        Assert.Contains("'1,234.5'", refusal, StringComparison.Ordinal);
        Assert.DoesNotContain("is read as text", refusal, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("N/A")]
    [InlineData("NA")]
    [InlineData("NaN")]
    [InlineData("?")]
    [InlineData("")]
    public void AFileAboutToBeScored_WithMissingNumbers_IsNotRefused(string missing)
        => Assert.Null(NumericLabelCheck.FileRefusal(Write("missing.csv", _ => missing), "y", "regression"));

    [Fact]
    public void ATaskNobodyKnows_HasNoLabelToAskAbout()
    {
        Assert.False(NumericLabelCheck.HasNumericLabel("nonsense"));
        Assert.Null(NumericLabelCheck.FileRefusal(Write("unknown.csv", _ => "words"), "y", "nonsense"));
    }

    [Fact]
    public void AFileAboutToBeScored_WhoseTaskNeedsNoNumberOrWhoseLabelIsAbsent_IsNotAskedAbout()
    {
        var path = Write("other.csv", _ => "unknown");

        Assert.Null(NumericLabelCheck.FileRefusal(path, "y", "multiclass-classification"));
        Assert.Null(NumericLabelCheck.FileRefusal(path, "no_such_column", "regression"));
        Assert.Null(NumericLabelCheck.FileRefusal(path, "", "regression"));
    }

    [Fact]
    public void AFileAboutToBeScored_WhoseLabelIsOnlyWords_IsRefusedAndPointsToClassification()
    {
        var path = Path.Combine(_tmpDir, "words.csv");
        File.WriteAllLines(path, ["a,y", "1,cat", "2,dog", "3,bird"]);

        var refusal = NumericLabelCheck.FileRefusal(path, "y", "regression");

        Assert.NotNull(refusal);
        Assert.Contains("3 of the 3 label values in the file", refusal, StringComparison.Ordinal);
        Assert.Contains("multiclass-classification", refusal, StringComparison.Ordinal);
    }

    [Fact]
    public void TheCensus_CountsNumbersMissingTokensAndUnreadableValuesApart()
    {
        var census = new NumericLabelCheck.Census();
        foreach (var cell in new[] { "1.5", " 2 ", "", "N/A", "?", "12%", "1,5", "x" })
            census.Add(cell);

        Assert.Equal(2, census.Numbers);
        Assert.Equal(7, census.Labelled);
        Assert.Equal(3, census.Unreadable);
        Assert.Equal(2, census.MissingTokens);
        Assert.Equal("12%", census.Example);
    }
}
