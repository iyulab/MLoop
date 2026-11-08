using System.Text;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.Commands;

/// <summary>
/// What <c>mloop train</c> does to a table before the search sees it: it says an empty or unclosed file is one, samples down to
/// <c>--max-rows</c>, drops the rows that have no label, and checks that the steps declared under <c>prep:</c> name columns the data has.
/// None of these ran in a test as a whole command — the branches were reached only by the helpers behind them, so
/// whether the command used the helper's answer (sampled file handed on, cleaned file handed on, refusal exits 1) was unobserved.
/// Every case stops after the preparation with <c>--analyze-data</c>, which is the command's own early exit, so no search runs.
/// </summary>
[Collection("FileSystem")]
public sealed class TrainPrepareDataTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-prepare-" + Guid.NewGuid().ToString("N"));

    public TrainPrepareDataTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private async Task<string> ProjectAsync(string task = "binary-classification", string? prepYaml = null)
    {
        Directory.SetCurrentDirectory(_workspace);
        var (exit, output, _) = await CliRunner.RunAsync("init", "p", "--task", task);
        Assert.True(exit == 0, output);
        var root = Path.Combine(_workspace, "p");
        Directory.SetCurrentDirectory(root);
        if (prepYaml is not null)
        {
            // The generated file names its label `Label`; the data below calls it `label`.
            // `mloop init` writes the platform's line endings (CRLF on Windows), so the line is matched either way.
            var yaml = await File.ReadAllTextAsync("mloop.yaml");
            var edited = System.Text.RegularExpressions.Regex.Replace(yaml, @"    label: Label\r?\n", "    label: label\n" + prepYaml);
            Assert.NotEqual(yaml, edited);   // the label line was found, so the step is in the file
            await File.WriteAllTextAsync("mloop.yaml", edited);
        }
        return root;
    }

    // x numbers the row; z is a number the row's prep step can name; the label alternates, and every `blankEvery`th row has none.
    private static IEnumerable<string> Rows(int count, int blankEvery = 0)
    {
        yield return "x,z,label";
        for (var i = 0; i < count; i++)
        {
            var label = blankEvery > 0 && i % blankEvery == 0 ? "" : i % 2 == 0 ? "yes" : "no";
            yield return $"{i},{(i * 7 % 11) / 10.0:0.0},{label}";
        }
    }

    // The console colours a count and its noun apart; a test reads the words.
    private static string Plain(string text) => System.Text.RegularExpressions.Regex.Replace(text, "\x1B\\[[0-9;]*m", "");

    private static async Task<(int ExitCode, string Stdout, string Stderr)> TrainAsync(params string[] extra)
    {
        var (exit, stdout, stderr) = await CliRunner.RunAsync([.. new[] { "train", "datasets/train.csv", "--label", "label", "--analyze-data" }, .. extra]);
        return (exit, Plain(stdout), Plain(stderr));
    }

    [Theory]
    [InlineData("", "is empty")]
    [InlineData("x,z,label\n", "has a header row and no data rows")]
    [InlineData("x,z,label\n1,\"abc,yes\n2,3,no\n", "is never closed")]
    public async Task AFileThatIsEmptyHeaderOnlyOrUnclosed_IsSaidSo_AndTheCommandStops(string content, string said)
    {
        await ProjectAsync();
        await File.WriteAllTextAsync("datasets/train.csv", content);

        var (exit, stdout, stderr) = await TrainAsync();

        Assert.Equal(1, exit);
        Assert.Contains(said, stderr + stdout);
        Assert.Contains("train on another file", stderr + stdout);
    }

    [Fact]
    public async Task MaxRows_HandsTheSampledFileOn_WithTheSeedAndStrategyItWasGiven()
    {
        await ProjectAsync();
        await File.WriteAllLinesAsync("datasets/train.csv", Rows(300));

        var (exit, stdout, _) = await TrainAsync("--max-rows", "100", "--seed", "7");

        Assert.Equal(0, exit);
        Assert.Contains("Sampled 300", stdout);
        Assert.Contains("stratified by label, seed=7", stdout);   // classification defaults to stratified; the seed reached the sampling
        var sampled = (await File.ReadAllLinesAsync("datasets/train_sampled.csv")).Skip(1).Where(l => l.Length > 0).ToArray();
        Assert.InRange(sampled.Length, 90, 100);   // never above the limit
        Assert.Contains("Class Distribution", stdout);   // the analysis went on, over the sampled rows
    }

    [Fact]
    public async Task AnExplicitRandomStrategy_IsTheOneThatRuns()
    {
        await ProjectAsync();
        await File.WriteAllLinesAsync("datasets/train.csv", Rows(300));

        var (exit, stdout, _) = await TrainAsync("--max-rows", "100", "--sampling-strategy", "random");

        Assert.Equal(0, exit);
        Assert.Contains("(random, seed=42)", stdout);
        Assert.Equal(100, (await File.ReadAllLinesAsync("datasets/train_sampled.csv")).Skip(1).Count(l => l.Length > 0));
    }

    [Fact]
    public async Task RowsWithNoLabel_AreDroppedByDefault_AndTheCleanedFileIsWhatIsAnalyzed()
    {
        await ProjectAsync();
        await File.WriteAllLinesAsync("datasets/train.csv", Rows(100, blankEvery: 5));   // 20 rows without a label

        var (exit, stdout, _) = await TrainAsync();

        Assert.Equal(0, exit);
        Assert.Contains("Dropped 20 rows with missing labels", stdout);
        var cleaned = (await File.ReadAllLinesAsync("datasets/train_cleaned.csv")).Skip(1).Where(l => l.Length > 0).ToArray();
        Assert.Equal(80, cleaned.Length);
        Assert.All(cleaned, row => Assert.NotEqual("", row.Split(',')[^1]));
    }

    [Fact]
    public async Task DropMissingLabelsFalse_KeepsTheRows_SoTheEmptyLabelIsAClassTheTaskRefuses()
    {
        await ProjectAsync();
        await File.WriteAllLinesAsync("datasets/train.csv", Rows(100, blankEvery: 5));

        var (exit, stdout, stderr) = await TrainAsync("--drop-missing-labels", "false");

        Assert.Equal(1, exit);   // yes, no and the empty label: three classes for a two-class task
        Assert.DoesNotContain("Dropped", stdout);
        Assert.Contains("holds 3 different values", stderr + stdout);
        Assert.False(File.Exists("datasets/train_cleaned.csv"));
    }

    [Fact]
    public async Task AFileWithNoMissingLabels_SaysTheLabelIsWhole_AndWritesNoCleanedFile()
    {
        await ProjectAsync();
        await File.WriteAllLinesAsync("datasets/train.csv", Rows(60));

        var (exit, stdout, _) = await TrainAsync();

        Assert.Equal(0, exit);
        Assert.Contains("has no empty values", stdout);
        Assert.False(File.Exists("datasets/train_cleaned.csv"));
    }

    [Fact]
    public async Task APrepStepThatNamesAColumnTheDataLacks_IsRefusedBeforeTheSearch_NamingTheStepAndTheColumns()
    {
        await ProjectAsync(prepYaml: "    prep:\n      - type: fill-missing\n        method: mean\n        columns: [nonexistent]\n");
        await File.WriteAllLinesAsync("datasets/train.csv", Rows(60));

        var (exit, stdout, stderr) = await TrainAsync();

        Assert.Equal(1, exit);
        var said = stderr + stdout;
        Assert.Contains("A prep step in mloop.yaml names a column that train.csv does not have: 'nonexistent'", said);
        Assert.Contains("Available columns: x, z, label", said);
        Assert.DoesNotContain("Class Distribution", said);   // nothing after the preparation ran
    }

    [Fact]
    public async Task APrepStepThatNamesAColumnInOtherLetters_IsToldTheDifferenceIsCase()
    {
        await ProjectAsync(prepYaml: "    prep:\n      - type: normalize\n        method: z-score\n        columns: [Z]\n");
        await File.WriteAllLinesAsync("datasets/train.csv", Rows(60));

        var (exit, stdout, stderr) = await TrainAsync();

        Assert.Equal(1, exit);
        Assert.Contains("'z' differs only in case", stderr + stdout);
    }

    [Fact]
    public async Task APrepStepThatNamesAColumnTheDataHas_PassesThePreparation()
    {
        await ProjectAsync(prepYaml: "    prep:\n      - type: normalize\n        method: z-score\n        columns: [z]\n");
        await File.WriteAllLinesAsync("datasets/train.csv", Rows(60));

        var (exit, stdout, _) = await TrainAsync();

        Assert.Equal(0, exit);
        Assert.Contains("Class Distribution", stdout);
    }

    [Fact]
    public void ThePrepColumnCheck_ReadsAKoreanHeaderInItsOwnEncoding()
    {
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        var path = Path.Combine(_workspace, "korean.csv");
        File.WriteAllText(path, "온도,압력,label\n1,2,yes\n", Encoding.GetEncoding(949));

        Assert.Null(TrainingInput.PrepColumnsRefusal(path, ["온도", "압력"]));
        var refusal = TrainingInput.PrepColumnsRefusal(path, ["습도"]);
        Assert.NotNull(refusal);
        Assert.Contains("'습도'", refusal);
        Assert.Contains("Available columns: 온도, 압력, label", refusal);
        Assert.DoesNotContain('�', refusal);
    }

    [Fact]
    public void ThePrepColumnCheck_HasNothingToSayAboutNoColumnsOrAFileItCannotRead()
    {
        var path = Path.Combine(_workspace, "t.csv");
        File.WriteAllText(path, "a,b\n1,2\n");

        Assert.Null(TrainingInput.PrepColumnsRefusal(path, []));
        Assert.Null(TrainingInput.PrepColumnsRefusal(Path.Combine(_workspace, "absent.csv"), ["a"]));
        Assert.Null(TrainingInput.PrepColumnsRefusal(Path.Combine(_workspace, "t.parquet"), ["nope"]));   // not a CSV or TSV
    }
}
