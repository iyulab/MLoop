using System.Text;
using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>train --max-rows</c> samples a large file down before training, and <c>--sampling-strategy</c> picks how. The sampling code
/// ran in no test: a file over the limit was cut, and nothing said the cut kept the classes' shares, the seed's sample, or a rare
/// class. A name other than <c>random</c> or <c>stratified</c> used to run as random and print "random" — a typo bought
/// the opposite of what it asked for. Now it is refused where the option is read and where the sampling is done.
/// </summary>
[Collection("FileSystem")]
public sealed class TrainSamplingTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-sampling-" + Guid.NewGuid().ToString("N"));

    public TrainSamplingTests()
    {
        Directory.CreateDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private string Write(string name, IEnumerable<string> lines, Encoding? encoding = null)
    {
        var path = Path.Combine(_workspace, name);
        File.WriteAllLines(path, lines, encoding ?? new UTF8Encoding(false));
        return path;
    }

    // x numbers every row, so a sample can be compared row by row; the label is the class the row belongs to.
    private static IEnumerable<string> Rows(params (string Label, int Count)[] classes)
    {
        yield return "x,label";
        var i = 0;
        foreach (var (label, count) in classes)
            for (var n = 0; n < count; n++)
                yield return $"{i++},{label}";
    }

    private static string[] DataLines(string path) => [.. File.ReadAllLines(path, Encoding.UTF8).Skip(1).Where(l => l.Length > 0)];

    private static Dictionary<string, int> ClassCounts(string path)
        => DataLines(path).GroupBy(l => l.Split(',')[^1]).ToDictionary(g => g.Key, g => g.Count());

    private Task<string> SampleAsync(string path, int maxRows, string task, string? label, string? strategy, int seed = 42)
        => TrainDataValidator.ApplySamplingIfNeededAsync(path, maxRows, task, label, strategy, seed, _workspace);

    [Fact]
    public async Task AFileWithinTheLimit_IsTrainedOnAsGiven()
    {
        var path = Write("small.csv", Rows(("a", 30)));

        var result = await SampleAsync(path, maxRows: 30, "binary-classification", "label", null);

        Assert.Equal(path, result);
        Assert.False(File.Exists(Path.Combine(_workspace, "train_sampled.csv")));
    }

    [Fact]
    public async Task RandomSampling_KeepsExactlyTheLimit_FromTheFile_AndTheSameSeedGivesTheSameRows()
    {
        var path = Write("big.csv", Rows(("a", 400)));

        var first = DataLines(await SampleAsync(path, 50, "regression", "label", "random", seed: 7));
        var again = DataLines(await SampleAsync(path, 50, "regression", "label", "random", seed: 7));
        var other = DataLines(await SampleAsync(path, 50, "regression", "label", "random", seed: 8));

        Assert.Equal(50, first.Length);
        Assert.Equal(50, first.Distinct().Count());
        Assert.Equal(first, again);
        Assert.NotEqual(first, other);
        Assert.All(first, row => Assert.InRange(int.Parse(row.Split(',')[0]), 0, 399));
    }

    [Fact]
    public async Task StratifiedSampling_KeepsEachClassesShare_AndNeverDropsARareClass()
    {
        // 300 rows cut to 100: a third of each class, rounded down, a class always keeping at least one row.
        var path = Write("imbalanced.csv", Rows(("common", 200), ("some", 80), ("few", 19), ("one", 1)));

        var counts = ClassCounts(await SampleAsync(path, 100, "multiclass-classification", "label", "stratified"));

        Assert.Equal(66, counts["common"]);
        Assert.Equal(26, counts["some"]);
        Assert.Equal(6, counts["few"]);
        Assert.Equal(1, counts["one"]);
    }

    [Theory]
    [InlineData("binary-classification", true)]
    [InlineData("multiclass-classification", true)]
    [InlineData("regression", false)]
    public async Task WithNoStrategyNamed_ClassificationIsStratified_AndOtherTasksAreRandom(string task, bool keepsShares)
    {
        var path = Write("default.csv", Rows(("a", 270), ("b", 30)));

        var rows = DataLines(await SampleAsync(path, 100, task, "label", strategy: null));
        var counts = rows.GroupBy(l => l.Split(',')[^1]).ToDictionary(g => g.Key, g => g.Count());

        if (keepsShares)
        {
            Assert.Equal(90, counts["a"]);   // exactly a third of each class
            Assert.Equal(10, counts["b"]);
        }
        else
        {
            Assert.Equal(100, rows.Length);   // random: the limit, not the shares
        }
    }

    [Fact]
    public async Task StratifiedWithNoLabelColumnToStratifyBy_SamplesRandomly()
    {
        var path = Write("nolabel.csv", Rows(("a", 200)));

        var rows = DataLines(await SampleAsync(path, 40, "binary-classification", label: null, "stratified"));

        Assert.Equal(40, rows.Length);
    }

    [Fact]
    public async Task ACp949File_IsSampledAsKorean_NotAsGarbage()
    {
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        var cp949 = Encoding.GetEncoding(949);
        var path = Write("korean.csv", Rows(("정상", 150), ("불량", 50)), cp949);

        var sampled = await SampleAsync(path, 40, "binary-classification", "label", "stratified");

        var counts = ClassCounts(sampled);
        Assert.Equal(30, counts["정상"]);
        Assert.Equal(10, counts["불량"]);
        Assert.DoesNotContain('�', File.ReadAllText(sampled, Encoding.UTF8));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("  ")]
    [InlineData("random")]
    [InlineData("Stratified")]
    [InlineData(" STRATIFIED ")]
    public void ANameSamplingKnows_OrNone_IsNotRefused(string? name)
        => Assert.Null(SamplingStrategies.Refusal(name));

    [Theory]
    [InlineData("bogus")]
    [InlineData("stratifed")]
    [InlineData("head")]
    [InlineData("random,stratified")]
    public void AnyOtherName_IsRefused_AndTheRefusalNamesTheChoices(string name)
    {
        var refusal = SamplingStrategies.Refusal(name);

        Assert.NotNull(refusal);
        Assert.Contains(name, refusal);
        foreach (var known in SamplingStrategies.Names)
            Assert.Contains(known, refusal);
    }

    [Fact]
    public async Task SamplingItself_RefusesAnUnknownName_EvenForAFileItWouldNotCut()
    {
        var path = Write("tiny.csv", Rows(("a", 5)));

        var error = await Assert.ThrowsAsync<ArgumentException>(() => SampleAsync(path, 100, "binary-classification", "label", "bogus"));

        Assert.Contains("Unknown sampling strategy", error.Message);
    }

    [Theory]
    [InlineData("random")]
    [InlineData("stratified")]
    public async Task EveryNameTheRefusalOffers_IsOneSamplingRuns(string name)
    {
        Assert.Contains(name, SamplingStrategies.Names);   // the data below must cover every name; a new one is added here too
        var path = Write($"each-{name}.csv", Rows(("a", 80), ("b", 40)));

        var rows = DataLines(await SampleAsync(path, 60, "binary-classification", "label", name));

        Assert.InRange(rows.Length, 58, 60);
    }

    [Theory]
    [InlineData("--sampling-strategy")]
    [InlineData("--sampling")]   // the hidden alias reads the same value
    public async Task Train_WithAnUnknownSamplingStrategy_FailsBeforeAnyWork(string option)
    {
        Directory.SetCurrentDirectory(_workspace);
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        File.WriteAllLines("datasets/train.csv", Rows(("yes", 60), ("no", 60)));

        var (exit, _, stderr) = await CliRunner.RunAsync(
            "train", "datasets/train.csv", "--label", "label", "--max-rows", "40", option, "stratifed", "--time", "5", "--max-trials", "3");   // refused before it trains; the trial count keeps a stray run bounded

        Assert.Equal(1, exit);
        Assert.Contains("Unknown sampling strategy 'stratifed'", stderr);
        Assert.Contains("random, stratified", stderr);
        Assert.False(File.Exists("datasets/train_sampled.csv"), "nothing was sampled");
    }
}
