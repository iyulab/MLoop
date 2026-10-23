using Microsoft.ML;
using MLoop.Core.Data;
using MLoop.Core.Training;
using MLoop.Search;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// A string column of codes or levels is a category (one-hot), a column of language is text — one
/// rule, decided from the whole file, read by featurization and by the saved schema alike.
/// </summary>
public sealed class CategoryColumnsTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "mloop-cat-" + Guid.NewGuid().ToString("N"));

    public CategoryColumnsTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    [Fact]
    public void Decide_PicksCodesAndLevels_NotLanguageNumbersOrSkippedColumns()
    {
        var rng = new Random(1);
        string[] levels = ["INFO", "WARN", "ERROR"];
        string[] sentences =
        [
            "the disk on the second volume failed during the nightly backup",
            "a user signed in from a new device and was asked to confirm",
            "the cache was refreshed after the configuration changed",
        ];
        var lines = new List<string> { "level,message,score,small_int,grade,label" };
        for (var i = 0; i < 300; i++)
            lines.Add($"{levels[i % 3]},\"{sentences[i % 3]} {i}\",{rng.NextDouble():F3},{i % 4},{(char)('A' + i % 5)},{(i % 2 == 0 ? "yes" : "no")}");
        var path = Path.Combine(_dir, "data.csv");
        File.WriteAllLines(path, lines);

        var decided = CategoryColumns.Decide(path, skip: ["label", "grade"]);

        Assert.Equal(["level"], decided); // message reads as language; numbers are numbers; grade was skipped
    }

    [Theory]
    [InlineData(new[] { "M", "F" }, 2, 1000, true)]
    [InlineData(new[] { "Bachelors", "HS-grad", "Masters" }, 16, 32000, true)]
    // Many distinct short codes, but under a tenth of the rows: still enumerable.
    [InlineData(new[] { "A1", "B2" }, 900, 30000, true)]
    // Many distinct codes and most rows distinct: an identifier-like column, not categories.
    [InlineData(new[] { "X1", "Y2" }, 900, 1000, false)]
    // Language: several words per value.
    [InlineData(new[] { "the quick brown fox", "jumps over the lazy dog" }, 5, 1000, false)]
    public void IsCategory_FollowsTextLikenessAndCardinality(string[] sample, int unique, int rows, bool expected)
        => Assert.Equal(expected, CategoryColumns.IsCategory(sample, unique, rows));

    [Fact]
    public void BuildColumnRoles_OneHotsTheDecidedCategories_AndTextsTheRest()
    {
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(new[]
        {
            new Row { Level = "INFO", Message = "backup completed without errors", X = 1, Label = true },
            new Row { Level = "WARN", Message = "disk almost full on the data volume", X = 2, Label = false },
        });

        var info = TrainingRunner.BuildColumnRoles(data, "Label", categoryColumns: ["Level"]);

        Assert.NotNull(info);
        Assert.Contains("Level", info.Of(ColumnRole.Categorical));
        Assert.DoesNotContain("Level", info.Of(ColumnRole.Text));
        Assert.Contains("Message", info.Of(ColumnRole.Text));
    }

    [Fact]
    public void BuildColumnRoles_ATypedColumnKeepsItsDeclaredType()
    {
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(new[]
        {
            new Row { Level = "INFO", Message = "m", X = 1, Label = true },
        });

        var info = TrainingRunner.BuildColumnRoles(data, "Label",
            columnOverrides: new() { ["Level"] = "text" }, categoryColumns: ["Level"]);

        Assert.Contains("Level", info.Of(ColumnRole.Text));
        Assert.DoesNotContain("Level", info.Of(ColumnRole.Categorical));
    }

    private sealed class Row
    {
        public string Level { get; set; } = "";
        public string Message { get; set; } = "";
        public float X { get; set; }
        public bool Label { get; set; }
    }
}
