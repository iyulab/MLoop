using System.Globalization;
using System.Text;
using MLoop.Core.Diagnostics;

namespace MLoop.Core.Tests.Diagnostics;

/// <summary>
/// A column that by itself nearly is the label. The shapes are the ones of the measurement behind the thresholds
/// (a class-code copy with 5 % of the rows flipped; the label plus noise at 10 % of its sd) and the ones that must stay quiet
/// (a genuinely strong column, an identifier, a mostly-constant column in a file sorted by label).
/// </summary>
public class SingleColumnLabelCheckTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "mloop-scl-" + Guid.NewGuid().ToString("N"));

    public SingleColumnLabelCheckTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private string Write(string header, IEnumerable<string> rows)
    {
        var path = Path.Combine(_dir, Guid.NewGuid().ToString("N") + ".csv");
        File.WriteAllText(path, header + "\n" + string.Join("\n", rows) + "\n", new UTF8Encoding(false));
        return path;
    }

    private static string Num(double v) => v.ToString("R", CultureInfo.InvariantCulture);

    private static IEnumerable<(int Cls, int Copy, string Noise, double X)> ClassRows(int n, int seed, double flip, bool sortedByLabel = false)
    {
        var rng = new Random(seed);
        var rows = Enumerable.Range(0, n).Select(i => (Cls: i % 3, i)).ToList();
        if (!sortedByLabel) rows = [.. rows.OrderBy(_ => rng.Next())];
        else rows = [.. rows.OrderBy(r => r.Cls)];
        foreach (var (cls, _) in rows)
        {
            var copy = rng.NextDouble() < flip ? (cls + 1 + rng.Next(2)) % 3 : cls;
            yield return (cls, copy, "n" + rng.Next(4), rng.NextDouble());
        }
    }

    [Fact]
    public void ACopyOfTheLabelWithFivePercentFlipped_IsNamed_AndNothingElse()
    {
        var path = Write("noise,x,target_copy,label",
            ClassRows(600, 1, 0.05).Select(r => $"{r.Noise},{Num(r.X)},{r.Copy},c{r.Cls}"));

        var found = SingleColumnLabelCheck.Find(path, "label", "multiclass-classification");

        var one = Assert.Single(found);
        Assert.Equal("target_copy", one.Column);
        Assert.False(one.IsRegression);
        Assert.InRange(one.Score, 0.92, 0.98);          // names the label for about 95 % of rows
        Assert.InRange(one.Baseline, 0.30, 0.37);       // three even classes: the most common alone gets a third
        Assert.True(one.Reduction >= SingleColumnLabelCheck.ClassificationThreshold);
    }

    [Fact]
    public void AGenuinelyStrongButNotNearCopyColumn_StaysQuiet()
    {
        // agrees with the label on 70 % of rows: strong, and not a copy
        var path = Write("strong,label",
            ClassRows(600, 2, 0.30).Select(r => $"{r.Copy},c{r.Cls}"));

        Assert.Empty(SingleColumnLabelCheck.Find(path, "label", "multiclass-classification"));
    }

    [Fact]
    public void AnIdentifierColumn_CannotLookLikeTheLabel()
    {
        var path = Write("account,label",
            ClassRows(600, 3, 0).Select((r, i) => $"acct-{i:D5},c{r.Cls}"));

        Assert.Empty(SingleColumnLabelCheck.Find(path, "label", "multiclass-classification"));
    }

    [Fact]
    public void AMostlyConstantColumn_InAFileSortedByTheLabel_IsNotTakenForTheLabel()
    {
        // Cutting a numeric column by RANK splits a run of equal values by row order; in a file sorted by label that order is the label.
        var rng = new Random(4);
        var rows = ClassRows(600, 4, 0, sortedByLabel: true)
            .Select((r, i) => $"{(rng.Next(100) == 0 ? "1" : "0")},{Num(r.X * 1000)},c{r.Cls}");
        var path = Write("rare,measure,label", rows);

        Assert.Empty(SingleColumnLabelCheck.Find(path, "label", "multiclass-classification"));
    }

    [Fact]
    public void ANumericLabelCopyWithNoise_IsNamed_AsAShareOfVariance()
    {
        var rng = new Random(5);
        var data = Enumerable.Range(0, 500).Select(_ => rng.NextDouble() * 100).ToList();
        var sd = Math.Sqrt(data.Average(v => Math.Pow(v - data.Average(), 2)));
        var rows = data.Select(y => $"{Num(rng.NextDouble())},{Num(y + (rng.NextDouble() - 0.5) * 0.35 * sd)},{Num(y)}");
        var path = Write("other,copy,price", rows);

        var one = Assert.Single(SingleColumnLabelCheck.Find(path, "price", "regression"));
        Assert.Equal("copy", one.Column);
        Assert.True(one.IsRegression);
        Assert.True(one.Score >= SingleColumnLabelCheck.RegressionThreshold);
    }

    [Fact]
    public void AnExactLinearCopyOfANumericLabel_IsNamed()
    {
        var rng = new Random(6);
        var rows = Enumerable.Range(0, 300).Select(_ => rng.NextDouble() * 50).Select(y => $"{Num(2 * y + 3)},{Num(y)}");
        var path = Write("twice,price", rows);

        Assert.Equal("twice", Assert.Single(SingleColumnLabelCheck.Find(path, "price", "regression")).Column);
    }

    [Fact]
    public void AStrongButNotCopyNumericColumn_StaysQuiet()
    {
        var rng = new Random(7);
        // r² about 0.8 between column and label
        var rows = Enumerable.Range(0, 500).Select(_ => rng.NextDouble() * 100)
            .Select(x => $"{Num(x)},{Num(x + (rng.NextDouble() - 0.5) * 100)}");
        var path = Write("x,y", rows);

        Assert.Empty(SingleColumnLabelCheck.Find(path, "y", "regression"));
    }

    [Fact]
    public void ColumnsThatAreNotFeatures_AreNotReported_AndNeitherIsTheLabel()
    {
        var path = Write("target_copy,label",
            ClassRows(600, 8, 0.02).Select(r => $"{r.Copy},c{r.Cls}"));

        Assert.Single(SingleColumnLabelCheck.Find(path, "label", "binary-classification", skip: []));
        Assert.Empty(SingleColumnLabelCheck.Find(path, "label", "binary-classification", skip: ["TARGET_COPY"]));
    }

    [Fact]
    public void ATableTooSmallToPredictFrom_ASingleClassLabel_AndATaskWithoutAFeatureLabel_AreNotJudged()
    {
        var small = Write("copy,label", ClassRows(60, 9, 0).Select(r => $"{r.Copy},c{r.Cls}"));
        var oneClass = Write("copy,label", Enumerable.Range(0, 300).Select(i => $"{i % 7},c0"));
        var normal = Write("copy,label", ClassRows(600, 9, 0).Select(r => $"{r.Copy},c{r.Cls}"));

        Assert.Empty(SingleColumnLabelCheck.Find(small, "label", "multiclass-classification"));
        Assert.Empty(SingleColumnLabelCheck.Find(oneClass, "label", "multiclass-classification"));
        Assert.Empty(SingleColumnLabelCheck.Find(normal, "label", "forecasting"));
        Assert.Empty(SingleColumnLabelCheck.Find(Path.Combine(_dir, "missing.csv"), "label", "regression"));
        Assert.Empty(SingleColumnLabelCheck.Find(normal, "no_such_column", "multiclass-classification"));
    }

    [Fact]
    public void RowsWithoutALabel_AreLeftOut_AndDoNotBreakTheCount()
    {
        var rows = ClassRows(600, 10, 0).Select(r => $"{r.Copy},c{r.Cls}").ToList();
        rows.AddRange(Enumerable.Repeat("1,", 40));
        var path = Write("copy,label", rows);

        var one = Assert.Single(SingleColumnLabelCheck.Find(path, "label", "multiclass-classification"));
        Assert.InRange(one.Score, 0.99, 1.0);
    }

    [Fact]
    public void LeaveOneOut_GivesAnIdentifierNothingToPredictFrom()
    {
        int n = 300;
        var classes = Enumerable.Range(0, n).Select(i => i % 3).ToArray();
        var identifier = Enumerable.Range(0, n).ToArray();           // one level per row
        var copy = classes.ToArray();                                // one level per class

        Assert.Equal(0, SingleColumnLabelCheck.LeaveOneOutReduction(identifier, classes).Lambda, 12);
        Assert.Equal(1, SingleColumnLabelCheck.LeaveOneOutReduction(copy, classes).Lambda, 12);
    }

    [Fact]
    public void LevelsOf_CutsANumericColumnByValue_SoEqualValuesShareABin()
    {
        var raw = Enumerable.Range(0, 400).Select(i => i < 394 ? "0" : (i - 393).ToString(CultureInfo.InvariantCulture)).ToArray();
        var (levels, numeric) = SingleColumnLabelCheck.LevelsOf(raw);

        Assert.NotNull(numeric);
        Assert.Single(levels.Take(394).Distinct());                  // the 394 zeros are one level whatever their row order
    }

    [Fact]
    public void ARowCap_ShrinksWithTheWidth_SoAWideTableIsNotHeldWhole()
    {
        Assert.Equal(SingleColumnLabelCheck.MaxRows, SingleColumnLabelCheck.RowCap(10));
        var wide = SingleColumnLabelCheck.RowCap(2049);
        Assert.InRange(wide, 400, 5000);
        Assert.True((long)wide * 2049 <= SingleColumnLabelCheck.MaxCells);
        Assert.Equal(400, SingleColumnLabelCheck.RowCap(1_000_000));   // never below what a column's levels can be predicted from
    }
}
