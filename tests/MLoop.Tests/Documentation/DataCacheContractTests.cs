using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// Rows kept in memory come from <c>RowCache.Of</c>, which fills the copy before anything reads it. ML.NET's own
/// <c>Data.Cache</c> fills in the background while the first cursor reads, and a reader that asks for a vector column
/// in that window can fail with "If specified, must be non-negative (Parameter 'valuesCount')" - at random, on large
/// tables, which is how a training run on millions of rows died half the time.
/// </summary>
public class DataCacheContractTests
{
    private static readonly Regex Direct = new(@"\.Data\s*\.\s*Cache\s*\(", RegexOptions.Compiled);

    private static IEnumerable<string> Product() =>
        RepoSourceTree.ProductionSourceFiles("src", "tools")
            .Where(f => !f.EndsWith("RowCache.cs", StringComparison.Ordinal));

    [Fact]
    public void NoProductCodeCachesRowsWithoutFillingThemFirst()
    {
        var offenders = Product()
            .Where(f => Direct.IsMatch(File.ReadAllText(f)))
            .Select(f => Path.GetRelativePath(RepoSourceTree.RepoRoot, f))
            .ToList();

        Assert.True(offenders.Count == 0,
            "Keep rows in memory with MLoop.Search.RowCache.Of, not MLContext.Data.Cache:"
            + Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }

    [Fact]
    public void TheScan_SeesTheCallWrittenAcrossLines_AndThePlacesWhereTheCacheIsLegitimate()
    {
        Assert.Matches(Direct, "x = _mlContext.Data.Cache(view);");
        Assert.Matches(Direct, "x = new MLContext(1)\n    .Data\n    .Cache(view);");
        // the one place that is allowed to call it is the one the scan skips - and it does call it
        var rowCache = RepoSourceTree.ProductionSourceFiles("src").Single(f => f.EndsWith("RowCache.cs", StringComparison.Ordinal));
        Assert.Matches(Direct, File.ReadAllText(rowCache));
        Assert.DoesNotContain(rowCache, Product());
        Assert.DoesNotMatch(Direct, "x = RowCache.Of(ml, view);");
        // and the scan reads the product tree at all
        Assert.True(Product().Count() > 100);
    }
}
