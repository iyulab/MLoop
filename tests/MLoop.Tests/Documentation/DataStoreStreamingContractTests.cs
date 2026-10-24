using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// The prediction-log and feedback stores read their day files a line at a time (<c>JsonLines.ReadAsync</c>). A
/// day file grows with traffic — 80,000 predictions a day is 20 MB — and the stores' readers once read it whole to
/// keep a few entries: <c>mloop logs --limit 5</c> held 282 MB, sampling 800,000 logged predictions 883 MB. A whole-file
/// read in the store is the shape that brings that back.
/// </summary>
public class DataStoreStreamingContractTests
{
    private static readonly Regex WholeFileRead = new(
        @"\bFile\.ReadAll(Lines|Text|Bytes)(Async)?\s*\(",
        RegexOptions.Compiled);

    private static IEnumerable<string> Store() =>
        RepoSourceTree.ProductionSourceFiles("src")
            .Where(f => f.Contains($"{Path.DirectorySeparatorChar}MLoop.DataStore{Path.DirectorySeparatorChar}", StringComparison.Ordinal));

    [Fact]
    public void TheStoresReadNoFileWhole()
    {
        var offenders = Store()
            .SelectMany(f => Offences(File.ReadAllText(f))
                .Select(line => $"{Path.GetRelativePath(RepoSourceTree.RepoRoot, f)}:{line}"))
            .ToList();

        Assert.True(offenders.Count == 0,
            "Read a store's day file with JsonLines.ReadAsync, keeping only what is needed:"
            + Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }

    [Fact]
    public void TheScan_SeesTheStoreSources() =>
        Assert.Contains(Store(), f => f.EndsWith("FilePredictionLogger.cs", StringComparison.Ordinal));

    private static IEnumerable<int> Offences(string source) =>
        WholeFileRead.Matches(source).Select(m => source.AsSpan(0, m.Index).Count('\n') + 1);

    [Theory]
    [InlineData("var lines = await File.ReadAllLinesAsync(filePath, cancellationToken).ConfigureAwait(false);")]
    [InlineData("foreach (var line in File.ReadAllLines(file))")]
    [InlineData("var json = File.ReadAllText (path);")]
    public void TheScan_SeesTheShapesItForbids(string line) =>
        Assert.NotEmpty(Offences(line));

    [Theory]
    [InlineData("await foreach (var line in JsonLines.ReadAsync(file, cancellationToken).ConfigureAwait(false))")]
    [InlineData("File.AppendAllText(filePath, json + Environment.NewLine);")]
    public void TheScan_PassesTheShapesItAllows(string line) =>
        Assert.Empty(Offences(line));
}
