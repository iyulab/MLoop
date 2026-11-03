using System.Globalization;
using System.Text;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.Commands;

/// <summary>
/// A table with two columns of one name is refused before the search, naming the columns and their positions. Before, the search
/// ran to its end and the run was lost at the last step with "An item with the same key has already been added".
/// </summary>
public class DuplicateColumnRefusalTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "mloop-dup-" + Guid.NewGuid().ToString("N"));

    public DuplicateColumnRefusalTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private string Write(string name, string text, Encoding? encoding = null)
    {
        var path = Path.Combine(_dir, name);
        File.WriteAllText(path, text, encoding ?? new UTF8Encoding(false));
        return path;
    }

    [Fact]
    public void TwoColumnsOfOneName_AreRefused_WithTheirPositions()
    {
        var path = Write("t.csv", "a,v,v,y\n1,2,3,yes\n");

        var refusal = TrainingInput.DuplicateColumnRefusal(path, "binary-classification");

        Assert.NotNull(refusal);
        Assert.Contains("'v' is columns 2, 3", refusal, StringComparison.Ordinal);
        Assert.Contains("t.csv", refusal, StringComparison.Ordinal);
    }

    [Fact]
    public void ThreeColumnsOfTwoNames_NameBoth()
    {
        var path = Write("t.csv", "a,v,v,w,w,y\n1,2,3,4,5,yes\n");

        var refusal = TrainingInput.DuplicateColumnRefusal(path, "regression")!;

        Assert.Contains("'v' is columns 2, 3", refusal, StringComparison.Ordinal);
        Assert.Contains("'w' is columns 4, 5", refusal, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("a,v,w,y\n1,2,3,yes\n")]                   // distinct
    [InlineData("a,A,y\n1,2,yes\n")]                       // names are case-sensitive: two names
    [InlineData(",a,,y\n0,1,2,yes\n")]                     // an index column has no name, and may be there twice
    public void DistinctNames_AreNotRefused(string text) =>
        Assert.Null(TrainingInput.DuplicateColumnRefusal(Write("ok.csv", text), "binary-classification"));

    [Fact]
    public void KoreanNamesInCp949_AreComparedAsKorean_NotAsReplacementCharacters()
    {
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        var cp949 = Encoding.GetEncoding(949);

        // two different names are not a pair, whatever their bytes look like when read as the wrong encoding
        Assert.Null(TrainingInput.DuplicateColumnRefusal(Write("ko.csv", "엑스,제트,결과\n1,2,예\n", cp949), "binary-classification"));
        var refusal = TrainingInput.DuplicateColumnRefusal(Write("ko2.csv", "엑스,제트,엑스,결과\n1,2,3,예\n", cp949), "binary-classification");
        Assert.NotNull(refusal);
        Assert.Contains("'엑스' is columns 1, 3", refusal, StringComparison.Ordinal);
    }

    [Fact]
    public void ANameBrokenOverTwoLines_IsOneName_AndTheSameWordInTwoCompositionsIsOne()
    {
        var composed = "한글";
        var decomposed = composed.Normalize(NormalizationForm.FormD);

        Assert.NotNull(TrainingInput.DuplicateColumnRefusal(Write("b.csv", "\"더블\n유\",x,\"더블 유\"\n1,2,3\n"), "regression"));
        Assert.NotNull(TrainingInput.DuplicateColumnRefusal(Write("c.csv", $"{composed},x,{decomposed}\n1,2,3\n"), "regression"));
        Assert.Null(TrainingInput.DuplicateColumnRefusal(Write("d.csv", "\"더블\n유\",x,\"더블 무\"\n1,2,3\n"), "regression"));
    }

    [Fact]
    public void ATabSeparatedFile_IsReadByTabs()
    {
        Assert.NotNull(TrainingInput.DuplicateColumnRefusal(Write("t.tsv", "a\tv\tv\ty\n1\t2\t3\tyes\n"), "regression"));
        // a comma inside a tab-separated name is not a separator
        Assert.Null(TrainingInput.DuplicateColumnRefusal(Write("u.tsv", "a,b\ta\ty\n1\t2\tyes\n"), "regression"));
    }

    [Fact]
    public void ADirectoryTask_AnotherFormat_OrAMissingFile_IsNotThisCheck()
    {
        Assert.Null(TrainingInput.DuplicateColumnRefusal(_dir, "image-classification"));
        Assert.Null(TrainingInput.DuplicateColumnRefusal(Write("t.json", "[]"), "regression"));
        Assert.Null(TrainingInput.DuplicateColumnRefusal(Path.Combine(_dir, "nope.csv"), "regression"));
        Assert.Null(TrainingInput.DuplicateColumnRefusal(Write("empty.csv", ""), "regression"));
    }

    [Fact]
    public async Task TheTrainCommand_RefusesBeforeAnyTrial()
    {
        var original = Directory.GetCurrentDirectory();
        try
        {
            Directory.SetCurrentDirectory(_dir);
            var init = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
            Assert.True(init.ExitCode == 0, init.Stdout + init.Stderr);
            Directory.SetCurrentDirectory(Path.Combine(_dir, "p"));

            var csv = new StringBuilder("a,v,v,y\n");
            var rng = new Random(4);
            for (var i = 0; i < 100; i++)
                csv.Append(CultureInfo.InvariantCulture, $"{rng.NextDouble():F3},{rng.NextDouble():F3},{rng.NextDouble():F3},{(i % 2 == 0 ? "yes" : "no")}\n");
            await File.WriteAllTextAsync("datasets/train.csv", csv.ToString());

            var (exit, stdout, stderr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "10", "--max-trials", "3");

            Assert.Equal(1, exit);
            var all = string.Join(' ', (stdout + " " + stderr).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
            Assert.Contains("'v' is columns 2, 3", all, StringComparison.Ordinal);
            Assert.Contains("Rename one of them", all, StringComparison.Ordinal);
            Assert.DoesNotContain("same key", all, StringComparison.Ordinal);
            Assert.False(Directory.Exists(Path.Combine("models", "default", "staging", "exp-001")), "no experiment was started");
        }
        finally
        {
            Directory.SetCurrentDirectory(original);
        }
    }
}
