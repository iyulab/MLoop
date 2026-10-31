using System.Globalization;
using System.Text;
using System.Text.Json;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>mloop sample create --from &lt;file&gt;</c> against what the guide promises: the sample is made of the file's own records (a field
/// with a comma, a quote or a line break stays one field), keeps their file order, holds each record once, takes the whole table when it
/// is asked for more rows than it has, gives every class a row in a stratified sample that has room, and is written beside the file
/// when no output is named. The text is written composed (NFC), as every command reads it.
/// </summary>
[Collection("FileSystem")]
public sealed class SampleFromFileTests : IDisposable
{
    private const string Composed = "한글";              // Hangul syllables
    private const string Decomposed = "한글"; // the same text as jamo

    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-sample-from-" + Guid.NewGuid().ToString("N"));

    public SampleFromFileTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private static readonly string[] Texts =
        ["plain", "has, comma", "say \"hi\"", "line1\nline2", Composed, Decomposed];

    private static string Quote(string field)
        => field.IndexOfAny([',', '"', '\n', '\r']) >= 0 ? "\"" + field.Replace("\"", "\"\"") + "\"" : field;

    /// <summary>300 records: classes a 70 %, b 25 %, c 4 %, d 1 %; the text column carries the awkward fields.</summary>
    private static List<string[]> SourceRecords()
    {
        var records = new List<string[]>();
        for (var i = 0; i < 300; i++)
        {
            var cls = i % 100 < 70 ? "a" : i % 100 < 95 ? "b" : i % 100 < 99 ? "c" : "d";
            records.Add([i.ToString(CultureInfo.InvariantCulture), cls, Texts[i % Texts.Length], i % 7 == 0 ? "" : "1.5"]);
        }
        return records;
    }

    private static async Task WriteSourceAsync(string path, List<string[]> records, string lineEnd = "\n", bool trailingBlankLine = false)
    {
        var text = new StringBuilder("id,cls,txt,val").Append(lineEnd);
        foreach (var r in records)
            text.Append(string.Join(',', r.Select(Quote))).Append(lineEnd);
        if (trailingBlankLine)
            text.Append(lineEnd);
        await File.WriteAllTextAsync(path, text.ToString(), new UTF8Encoding(false));
    }

    /// <summary>The records of a CSV file, a quoted field kept whole.</summary>
    private static List<string[]> Parse(string path)
    {
        var text = File.ReadAllText(path);
        var records = new List<string[]>();
        var fields = new List<string>();
        var field = new StringBuilder();
        var quoted = false;
        for (var i = 0; i < text.Length; i++)
        {
            var c = text[i];
            if (quoted)
            {
                if (c == '"' && i + 1 < text.Length && text[i + 1] == '"') { field.Append('"'); i++; }
                else if (c == '"') quoted = false;
                else field.Append(c);
            }
            else if (c == '"') quoted = true;
            else if (c == ',') { fields.Add(field.ToString()); field.Clear(); }
            else if (c == '\r') { }
            else if (c == '\n') { fields.Add(field.ToString()); field.Clear(); records.Add([.. fields]); fields.Clear(); }
            else field.Append(c);
        }
        if (field.Length > 0 || fields.Count > 0) { fields.Add(field.ToString()); records.Add([.. fields]); }
        return records;
    }

    private static string[] Composedly(string[] record) => [.. record.Select(f => f.Normalize(NormalizationForm.FormC))];

    private async Task<(int Sampled, int Total, List<string[]> Records)> SampleAsync(string source, string output, params string[] options)
    {
        var args = new List<string> { "sample", "create", "--from", source, "-o", output, "--json" };
        args.AddRange(options);
        var run = await CliRunner.RunAsync([.. args]);
        Assert.True(run.ExitCode == 0, run.Stdout + run.Stderr);
        using var json = JsonDocument.Parse(run.Stdout);
        var rows = Parse(output);
        Assert.Equal(["id", "cls", "txt", "val"], rows[0]);
        return (json.RootElement.GetProperty("sampled").GetInt32(), json.RootElement.GetProperty("totalRows").GetInt32(), rows[1..]);
    }

    [Theory]
    [InlineData("\n", false)]
    [InlineData("\r\n", false)]
    [InlineData("\n", true)]
    public async Task EveryStrategy_SamplesTheSourcesOwnRecords_InFileOrder_AndCountsOnlyRecords(string lineEnd, bool trailingBlankLine)
    {
        var source = SourceRecords();
        await WriteSourceAsync("source.csv", source, lineEnd, trailingBlankLine);
        var expected = source.Select(Composedly).ToList();
        var byId = expected.ToDictionary(r => r[0]);

        var head = await SampleAsync("source.csv", "head.csv", "--rows", "20", "--strategy", "head");
        Assert.Equal((20, 300), (head.Sampled, head.Total));
        Assert.Equal(expected.Take(20), head.Records);

        var picks = new List<HashSet<string>>();
        foreach (var run in new[] { "random-1.csv", "random-2.csv" })
        {
            var random = await SampleAsync("source.csv", run, "--rows", "20", "--strategy", "random");
            Assert.Equal((20, 300), (random.Sampled, random.Total));
            Assert.All(random.Records, r => Assert.Equal(byId[r[0]], r));
            var ids = random.Records.Select(r => int.Parse(r[0], CultureInfo.InvariantCulture)).ToList();
            Assert.Equal(ids.Order().Distinct(), ids);          // once each, in file order
            picks.Add([.. random.Records.Select(r => r[0])]);
        }

        var stratified = await SampleAsync("source.csv", "strat.csv", "--rows", "20", "--strategy", "stratified", "--label", "cls");
        Assert.Equal(20, stratified.Sampled);
        Assert.All(stratified.Records, r => Assert.Equal(byId[r[0]], r));
        var perClass = stratified.Records.GroupBy(r => r[1]).ToDictionary(g => g.Key, g => g.Count());
        Assert.Equal(["a", "b", "c", "d"], perClass.Keys.Order());      // every class has a row
        Assert.InRange(perClass["a"], 13, 14);                          // the large ones keep their share (70 % of 20), less the row a rare class took
        Assert.InRange(perClass["b"], 4, 5);

        foreach (var strategy in new[] { "head", "random", "stratified" })
        {
            var all = await SampleAsync("source.csv", $"all-{strategy}.csv", "--rows", "5000", "--strategy", strategy, "--label", "cls");
            Assert.Equal((300, 300), (all.Sampled, all.Total));
            Assert.Equal(expected, all.Records);
        }
    }

    [Fact]
    public async Task TheTextIsWrittenComposed_AndAnAwkwardFieldStaysOneField()
    {
        await WriteSourceAsync("source.csv", SourceRecords());
        var all = await SampleAsync("source.csv", "all.csv", "--rows", "300", "--strategy", "head");

        // the jamo spelling comes out as syllables - the same text, the way every command reads it
        Assert.DoesNotContain(all.Records, r => string.Equals(r[2], Decomposed, StringComparison.Ordinal));
        Assert.Equal(100, all.Records.Count(r => string.Equals(r[2], Composed, StringComparison.Ordinal)));
        Assert.Equal(50, all.Records.Count(r => r[2] == "line1\nline2"));
        Assert.Equal(50, all.Records.Count(r => r[2] == "say \"hi\""));
        Assert.Equal(50, all.Records.Count(r => r[2] == "has, comma"));
        Assert.Equal(43, all.Records.Count(r => r[3] == ""));          // empty cells stay empty
    }

    [Fact]
    public async Task WithoutAnOutput_TheSampleIsWrittenBesideTheFile()
    {
        Directory.CreateDirectory("data");
        await WriteSourceAsync(Path.Combine("data", "source.csv"), SourceRecords());
        var run = await CliRunner.RunAsync("sample", "create", "--from", Path.Combine("data", "source.csv"), "--rows", "10", "--strategy", "head");
        Assert.True(run.ExitCode == 0, run.Stdout + run.Stderr);
        Assert.Equal(10, Parse(Path.Combine("data", "source_sampled.csv")).Count - 1);
    }

    [Theory]
    [InlineData("--rows", "0")]
    [InlineData("--rows", "-5")]
    [InlineData("--strategy", "stratified")]
    public async Task ARequestItCannotHonour_FailsWithoutWritingAFile(string option, string value)
    {
        await WriteSourceAsync("source.csv", SourceRecords());
        var run = await CliRunner.RunAsync("sample", "create", "--from", "source.csv", "-o", "out.csv", option, value);
        Assert.NotEqual(0, run.ExitCode);
        Assert.False(File.Exists("out.csv"));
    }
}
