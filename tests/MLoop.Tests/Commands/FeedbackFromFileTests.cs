using System.Text;
using System.Text.Json;
using MLoop.Tests.Documentation;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>mloop feedback add --from-file</c>: feedback for many predictions from a table — one row per prediction, its id and the actual
/// value in named columns. The project is the one the command-runs guard builds (a promoted model and four hundred logged predictions),
/// so the ids are real log ids.
/// </summary>
[Collection("FileSystem")]
public sealed class FeedbackFromFileTests : IClassFixture<EveryCommandRunsContractTests.TrainedProject>
{
    private readonly EveryCommandRunsContractTests.TrainedProject _project;

    public FeedbackFromFileTests(EveryCommandRunsContractTests.TrainedProject project) => _project = project;

    /// <summary>
    /// Ids of logged predictions that no spreadsheet turns into a number (a hex id of digits only, or one like 1e5…, is read as one) —
    /// a different slice of the log for each test, so what one test records is not another's to count.
    /// </summary>
    private List<string> LoggedIds(int skip, int count)
    {
        var logs = Directory.EnumerateFiles(Path.Combine(_project.Root, ".mloop", "logs"), "*.jsonl", SearchOption.AllDirectories);
        return [.. logs.SelectMany(File.ReadLines)
            .Select(line => JsonDocument.Parse(line).RootElement.GetProperty("id").GetString()!)
            .Where(id => !double.TryParse(id, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out _))
            .Skip(skip).Take(count)];
    }

    private string Write(string name, string text)
    {
        File.WriteAllText(Path.Combine(_project.Root, name), text, new UTF8Encoding(false));
        return name;
    }

    private static string Table(string header, IEnumerable<string> rows) => header + "\n" + string.Join('\n', rows) + "\n";

    [Fact]
    public async Task RecordsEveryRowOfTheFileAndCountsThem()
    {
        var ids = LoggedIds(0, 20);
        var file = Write("fb-csv.csv", Table("prediction_id,actual", ids.Select(id => $"{id},yes")));

        var seen = await _project.RunAsync("feedback", "add", "--from-file", file, "--json");

        Assert.True(seen.Exit == 0, seen.Out + seen.Err);
        var json = seen.Json();
        Assert.Equal(20, json.GetProperty("rows").GetInt32());
        Assert.Equal(20, json.GetProperty("recorded").GetInt32());
        Assert.Empty(json.GetProperty("rejected").EnumerateArray());
        var listed = (await _project.RunAsync("feedback", "list", "--limit", "1000", "--json")).Json();
        var recorded = listed.EnumerateArray().Select(e => e.GetProperty("predictionId").GetString()).ToHashSet();
        Assert.All(ids, id => Assert.Contains(id, recorded));
    }

    [Fact]
    public async Task ReadsAWorkbookAsTheTableItHolds()
    {
        var ids = LoggedIds(20, 12);
        var workbook = Path.Combine(_project.Root, "fb-book.xlsx");
        WorkbookFile.Write(workbook, ["prediction_id", "actual"], ids.Select(id => new object[] { id, "no" }));

        var seen = await _project.RunAsync("feedback", "add", "--from-file", "fb-book.xlsx", "--json");

        Assert.True(seen.Exit == 0, seen.Out + seen.Err);
        var json = seen.Json();
        Assert.Equal(12, json.GetProperty("rows").GetInt32());
        Assert.Equal(12, json.GetProperty("recorded").GetInt32());
    }

    [Fact]
    public async Task ReadsAJsonLinesFileAsTheTableItHolds()
    {
        // The command takes whatever table reader the other commands take; a FilePrepper that does not read JSON Lines yet cannot be asked.
        Assert.SkipUnless(FilePrepper.Utils.DataFileFormats.IsReadable("labels.jsonl"), "the FilePrepper in use does not read JSON Lines");
        var ids = LoggedIds(70, 6);
        var file = Write("fb-lines.jsonl", string.Join('\n', ids.Select(id => $"{{\"prediction_id\":\"{id}\",\"actual\":\"yes\"}}")) + "\n");

        var seen = await _project.RunAsync("feedback", "add", "--from-file", file, "--json");

        Assert.True(seen.Exit == 0, seen.Out + seen.Err);
        Assert.Equal(6, seen.Json().GetProperty("recorded").GetInt32());
    }

    [Fact]
    public async Task ReadsASemicolonFileAsTheTableItHolds()
    {
        var ids = LoggedIds(32, 9);
        var file = Write("fb-semi.csv", Table("prediction_id;actual", ids.Select(id => $"{id};yes")));

        var seen = await _project.RunAsync("feedback", "add", "--from-file", file, "--json");

        Assert.True(seen.Exit == 0, seen.Out + seen.Err);
        Assert.Equal(9, seen.Json().GetProperty("recorded").GetInt32());
    }

    [Fact]
    public async Task ARowItCannotRecordIsNamedByItsRowAndTheExitIsNotZero()
    {
        var ids = LoggedIds(41, 4);
        var file = Write("fb-mixed.csv", Table("prediction_id,actual,weight",
        [
            $"{ids[0]},yes,",
            "0123456789abcdef,yes,",
            $"{ids[1]},,",
            $"{ids[2]},no,heavy",
            $"{ids[3]},no,-3",
        ]));

        var seen = await _project.RunAsync("feedback", "add", "--from-file", file, "--json");

        Assert.Equal(1, seen.Exit);
        var json = seen.Json();
        Assert.Equal(5, json.GetProperty("rows").GetInt32());
        Assert.Equal(1, json.GetProperty("recorded").GetInt32());
        var rejected = json.GetProperty("rejected").EnumerateArray().ToList();
        Assert.Equal([2, 3, 4, 5], rejected.Select(r => r.GetProperty("row").GetInt32()));
        Assert.Contains("no logged prediction", rejected[0].GetProperty("reason").GetString());
        Assert.Contains("actual value is empty", rejected[1].GetProperty("reason").GetString());
        Assert.Contains("'heavy' is not a number", rejected[2].GetProperty("reason").GetString());
        Assert.Contains("cannot weight a row", rejected[3].GetProperty("reason").GetString());

        // the same file in the terminal's words: the rows by number and the way to find an id
        var text = await _project.RunAsync("feedback", "add", "--from-file", file);
        Assert.Equal(1, text.Exit);
        Assert.Contains("row 2", text.Out + text.Err);
        Assert.Contains("mloop logs", text.Out + text.Err);
        Assert.DoesNotContain("Exception", text.Out + text.Err);
    }

    [Fact]
    public async Task TheColumnsAreNamedNotGuessed()
    {
        var ids = LoggedIds(45, 3);
        var file = Write("fb-names.csv", Table("ticket,verdict,how", ids.Select(id => $"{id},yes,reviewer-llm")));

        var refused = await _project.RunAsync("feedback", "add", "--from-file", file);

        Assert.Equal(1, refused.Exit);
        var message = refused.Out + refused.Err;
        Assert.Contains("fb-names.csv", message);
        Assert.Contains("'prediction_id'", message);
        Assert.Contains("ticket, verdict, how", message);
        Assert.Contains("--id-column", message);

        var seen = await _project.RunAsync("feedback", "add", "--from-file", file, "--id-column", "ticket", "--actual-column", "VERDICT",
            "--source-column", "how", "--json");

        Assert.True(seen.Exit == 0, seen.Out + seen.Err);
        Assert.Equal(3, seen.Json().GetProperty("recorded").GetInt32());
        var listed = (await _project.RunAsync("feedback", "list", "--limit", "1000", "--json")).Json().EnumerateArray()
            .Where(e => ids.Contains(e.GetProperty("predictionId").GetString()!)).ToList();
        Assert.All(listed, e => Assert.Equal("reviewer-llm", e.GetProperty("source").GetString()));
    }

    [Fact]
    public async Task ANamedColumnTheFileLacksIsAMistakeButAMissingDefaultIsNot()
    {
        var ids = LoggedIds(48, 2);
        var file = Write("fb-plain.csv", Table("prediction_id,actual", ids.Select(id => $"{id},yes")));

        var ok = await _project.RunAsync("feedback", "add", "--from-file", file, "--json");
        var named = await _project.RunAsync("feedback", "add", "--from-file", file, "--weight-column", "importance");

        Assert.True(ok.Exit == 0, ok.Out + ok.Err);
        Assert.Equal(1, named.Exit);
        Assert.Contains("'importance'", named.Out + named.Err);
        Assert.Contains("--weight-column", named.Out + named.Err);
    }

    [Fact]
    public async Task SourceAndWeightFillTheRowsWhoseCellsAreEmpty()
    {
        var ids = LoggedIds(50, 3);
        var file = Write("fb-defaults.csv", Table("prediction_id,actual,source,weight",
        [
            $"{ids[0]},yes,,",
            $"{ids[1]},yes,human,4",
            $"{ids[2]},no,,",
        ]));

        var seen = await _project.RunAsync("feedback", "add", "--from-file", file, "--source", "reviewer-llm", "--weight", "2", "--json");

        Assert.True(seen.Exit == 0, seen.Out + seen.Err);
        var listed = (await _project.RunAsync("feedback", "list", "--limit", "1000", "--json")).Json().EnumerateArray()
            .Where(e => ids.Contains(e.GetProperty("predictionId").GetString()!))
            .ToDictionary(e => e.GetProperty("predictionId").GetString()!);
        Assert.Equal(("reviewer-llm", 2.0), (listed[ids[0]].GetProperty("source").GetString(), listed[ids[0]].GetProperty("weight").GetDouble()));
        Assert.Equal(("human", 4.0), (listed[ids[1]].GetProperty("source").GetString(), listed[ids[1]].GetProperty("weight").GetDouble()));
        Assert.Equal(("reviewer-llm", 2.0), (listed[ids[2]].GetProperty("source").GetString(), listed[ids[2]].GetProperty("weight").GetDouble()));
    }

    [Fact]
    public async Task APredictionNamedTwiceStandsAsItsLastRow()
    {
        var id = LoggedIds(53, 1).Single();
        var file = Write("fb-twice.csv", Table("prediction_id,actual", [$"{id},wrong-label", $"{id},yes"]));

        var seen = await _project.RunAsync("feedback", "add", "--from-file", file, "--json");

        Assert.True(seen.Exit == 0, seen.Out + seen.Err);
        var json = seen.Json();
        Assert.Equal(2, json.GetProperty("recorded").GetInt32());
        var repeated = Assert.Single(json.GetProperty("repeated").EnumerateArray());
        Assert.Equal([1, 2], repeated.GetProperty("rows").EnumerateArray().Select(r => r.GetInt32()));
        Assert.NotEmpty(json.GetProperty("warnings").EnumerateArray());
        // the feedback list shows the newest entry first: the later row
        var entries = (await _project.RunAsync("feedback", "list", "--limit", "1000", "--json")).Json().EnumerateArray()
            .Where(e => e.GetProperty("predictionId").GetString() == id).ToList();
        Assert.Equal("yes", entries.First().GetProperty("actualValue").ToString());
    }

    [Theory]
    [InlineData("--prediction-id", "abc")]
    [InlineData("--actual-value", "yes")]
    public async Task OneRowAndAFileAreNotMixed(string option, string value)
    {
        var seen = await _project.RunAsync("feedback", "add", "--from-file", "fb-csv.csv", option, value);

        Assert.Equal(1, seen.Exit);
        Assert.Contains("--from-file cannot be combined", seen.Out + seen.Err);
    }

    [Fact]
    public async Task ColumnOptionsNeedTheFile()
    {
        var seen = await _project.RunAsync("feedback", "add", "--prediction-id", "abc", "--actual-value", "yes", "--id-column", "ticket");

        Assert.Equal(1, seen.Exit);
        Assert.Contains("--from-file", seen.Out + seen.Err);
    }

    [Fact]
    public async Task WithoutAFileBothOptionsOfOneRowAreStillRequired()
    {
        var seen = await _project.RunAsync("feedback", "add", "--prediction-id", "abc");

        Assert.Equal(1, seen.Exit);
        var message = seen.Out + seen.Err;
        Assert.Contains("--actual-value", message);
        Assert.Contains("--from-file", message);
    }

    [Theory]
    [InlineData("missing.csv", "not found")]
    public async Task AFileThatIsNotThereIsNamed(string file, string cause)
    {
        var seen = await _project.RunAsync("feedback", "add", "--from-file", file);

        Assert.Equal(1, seen.Exit);
        Assert.Contains(file, seen.Out + seen.Err);
        Assert.Contains(cause, seen.Out + seen.Err);
    }

    [Fact]
    public async Task AFileWithAHeaderAndNoRowsIsSaidSo()
    {
        var file = Write("fb-header-only.csv", "prediction_id,actual\n");

        var seen = await _project.RunAsync("feedback", "add", "--from-file", file);

        Assert.Equal(1, seen.Exit);
        Assert.Contains("no rows", seen.Out + seen.Err);
    }
}
