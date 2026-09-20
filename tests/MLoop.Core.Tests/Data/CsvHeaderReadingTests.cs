using System.Text;
using MLoop.Core.Data;
using MLoop.Core.Prediction;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// The header reader feeds <see cref="CsvDataLoader.DetermineExcludedColumns"/>, which computes its
/// entire answer by diffing header lists across the removal chain. A header read wrongly there does
/// not surface as a wrong list — it surfaces as an exclusion whose name matches no real column, and
/// the caller that applies exclusions skips those without a word.
/// </summary>
public class CsvHeaderReadingTests : IDisposable
{
    private readonly List<string> _temp = [];

    private string Write(string content, Encoding encoding)
    {
        var path = Path.Combine(Path.GetTempPath(), $"mloop-hdr-{Guid.NewGuid():N}.csv");
        File.WriteAllText(path, content, encoding);
        _temp.Add(path);
        return path;
    }

    public void Dispose()
    {
        foreach (var p in _temp)
        {
            try { File.Delete(p); } catch { /* best effort */ }
        }
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// The shape the naive split got wrong: a quoted header containing a comma is one column, and
    /// every other reader in the product — and ML.NET — reads it that way.
    /// </summary>
    [Fact]
    public void AQuotedCommaIsOneColumn()
    {
        var path = Write("\"Last, First\",Age,City\n\"Doe, Jane\",31,Seoul\n", new UTF8Encoding(true));

        var excluded = CsvDataLoader.DetermineExcludedColumns(path, labelColumn: "Age");

        // Nothing here is excludable; what matters is that the reader agreed with the parser about
        // the column count, which the exclusion diff depends on.
        Assert.Equal(
            ["Last, First", "Age", "City"],
            CsvFieldParser.ParseFields(File.ReadLines(path).First()));
        Assert.DoesNotContain(excluded, c => c.Name is "Last" or " First\"");
    }

    /// <summary>
    /// The quoted-comma column is itself the one dropped, so the name the authority reports is the
    /// name a caller has to match against the real schema. The naive split reported two fragments
    /// here, neither of which is a column, and the caller skipped both in silence.
    /// </summary>
    [Fact]
    public void AnExcludedColumnIsNamedAsTheSchemaNamesIt()
    {
        var rows = string.Join('\n',
        [
            "Id,\"Last, First\",Score",
            "1,\"Doe, Jane\",1",
            "2,\"Doe, Jane\",2",
            "3,\"Doe, Jane\",3",
        ]);
        var path = Write(rows + "\n", new UTF8Encoding(true));

        var excluded = CsvDataLoader.DetermineExcludedColumns(path, labelColumn: "Score");

        Assert.Contains(excluded, c => c.Name == "Last, First");
        Assert.DoesNotContain(excluded, c => c.Name is "Last" or "First");
    }

    /// <summary>
    /// Korean headers in CP949 — the encoding this product converts for everywhere else. Read as
    /// UTF-8 they come back mojibake, and every exclusion name derived from them matches nothing.
    /// </summary>
    [Fact]
    public void KoreanHeadersSurviveALegacyEncoding()
    {
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        var cp949 = Encoding.GetEncoding(949);

        var rows = string.Join('\n', ["고객명,지역,점수", "김철수,서울,1", "이영희,서울,2", "박민수,서울,3"]);
        var path = Write(rows + "\n", cp949);

        var excluded = CsvDataLoader.DetermineExcludedColumns(path, labelColumn: "점수");

        // "지역" is constant, so it is dropped — under its real name, not a mangled one.
        Assert.Contains(excluded, c => c.Name == "지역");
    }
}
