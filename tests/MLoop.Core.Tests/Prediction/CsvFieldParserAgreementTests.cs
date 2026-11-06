using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;
using MLoop.Core.Prediction;

namespace MLoop.Core.Tests.Prediction;

/// <summary>
/// The hand-written line splitter reads a line as the readers that follow it do (CsvHelper as the product configures it, and
/// ML.NET's TextLoader, which takes a quote as text unless it starts a field). It had its own rule - toggle on every quote - and the
/// two parted ways on text with a quote in it: 40 rows in, 36 predictions out. A line here is read both ways and must come out alike.
/// </summary>
public class CsvFieldParserAgreementTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"mloop-fieldagree-{Guid.NewGuid():N}");

    public CsvFieldParserAgreementTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        if (Directory.Exists(_dir))
            Directory.Delete(_dir, true);
    }

    public static TheoryData<string> Lines => new()
    {
        "a,b,c",
        "12\" pipe,11,1",
        "say \"hi\" now,3,1",
        "5\",3,1",
        "\"a, b\",c,d",
        "a,  \"b, c\",d",
        "\"say \"\"hi\"\"\",x",
        "\"\",x,y",
        "a,,c",
        "x,\"\",z",
        "한글 \"따옴표\" 포함,1,2",
        "a,b,",
    };

    [Theory]
    [MemberData(nameof(Lines))]
    public void ALineIsSplitAsTheReadersSplitIt(string line)
    {
        var path = Path.Combine(_dir, "line.csv");
        File.WriteAllText(path, line + "\n");
        var configuration = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            HasHeaderRecord = false,
            MissingFieldFound = null,
            BadDataFound = null,
            TrimOptions = TrimOptions.Trim,
        };
        using var reader = new StreamReader(path);
        using var csv = new CsvReader(reader, configuration);
        Assert.True(csv.Read());

        Assert.Equal(csv.Parser.Record!, CsvFieldParser.ParseFields(line));
    }

    [Fact]
    public void TheCorpusIsSeenToHoldTheShapesItIsForTheOldRuleSplitWrong()
    {
        // The guard asserts that it can see: under the old toggle-on-every-quote rule these lines come out as one field.
        var oldRule = (string line) =>
        {
            var fields = 1;
            var inQuotes = false;
            foreach (var c in line)
            {
                if (c == '"') inQuotes = !inQuotes;
                else if (c == ',' && !inQuotes) fields++;
            }
            return fields;
        };

        Assert.NotEqual(CsvFieldParser.ParseFields("12\" pipe,11,1").Length, oldRule("12\" pipe,11,1"));
        Assert.Equal(CsvFieldParser.ParseFields("a,b,c").Length, oldRule("a,b,c"));
    }
}
