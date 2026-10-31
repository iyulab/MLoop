using System.Text;
using Microsoft.ML;
using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// A file whose lines do not all have the header's number of fields does not split into columns, and
/// "check for a consistent number of columns" left the reader to find the line in a file of thousands. The
/// refusal now names the first such line. It stays a refusal with the same opening words —
/// <c>CsvDataLoader</c> recognises it by them — and a line that is only part of a quoted field spread over
/// several lines is not mistaken for a short row.
/// </summary>
public sealed class ColumnLayoutRaggedLineTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"mloop_ragged_{Guid.NewGuid():N}");

    public ColumnLayoutRaggedLineTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private string Write(Func<int, string> row, int rows = 40)
    {
        var text = new StringBuilder("a,b,c,label\n");
        for (var i = 1; i <= rows; i++)
            text.Append(row(i)).Append('\n');
        var path = Path.Combine(_dir, $"{Guid.NewGuid():N}.csv");
        File.WriteAllText(path, text.ToString(), new UTF8Encoding(encoderShouldEmitUTF8Identifier: true));
        return path;
    }

    private static string Row(int i) => $"{i},{i % 5},{i * 2},{(i % 3 == 0 ? "B" : "A")}";

    private static ColumnLayoutException Refusal(string path)
        => Assert.Throws<ColumnLayoutException>(() => ColumnLayoutInference.Infer(new MLContext(seed: 1), path, "label"));

    [Fact]
    public void AShortRow_IsNamedByItsLine()
    {
        var refusal = Refusal(Write(i => i == 9 ? "9,4,18" : Row(i)));

        Assert.StartsWith("Unable to split the file provided into multiple, consistent columns.", refusal.Message);
        Assert.Contains("Line 10 has 3 field(s) where the header has 4.", refusal.Message);
    }

    [Fact]
    public void ALongRow_IsNamedByItsLine()
    {
        var refusal = Refusal(Write(i => i == 17 ? Row(i) + ",extra" : Row(i)));

        Assert.Contains("Line 18 has 5 field(s) where the header has 4.", refusal.Message);
    }

    [Fact]
    public void OnlyTheFirstOffendingLineIsNamed()
    {
        var refusal = Refusal(Write(i => i is 5 or 30 ? "x,y" : Row(i)));

        Assert.Contains("Line 6 has 2 field(s)", refusal.Message);
        Assert.DoesNotContain("Line 31", refusal.Message);
    }

    [Fact]
    public void AQuotedFieldOverSeveralLines_IsNotMistakenForAShortRow()
    {
        // The layout reads this file (ReadMultilines); the hint must not fire on the lines of the cell either.
        var path = Write(i => i == 7 ? "7,2,14,\"A\nstill the label of row 7\"" : Row(i));

        var layout = ColumnLayoutInference.Infer(new MLContext(seed: 1), path, "label");

        Assert.NotEmpty(layout.Options.Columns);
    }

    [Fact]
    public void ABlankLine_IsNotARow_AndIsNotNamed()
    {
        var refusal = Refusal(Write(i => i == 3 ? "" : i == 20 ? "20,0" : Row(i)));

        Assert.Contains("Line 21 has 2 field(s)", refusal.Message);
    }
}
