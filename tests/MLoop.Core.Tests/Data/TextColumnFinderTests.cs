using Microsoft.ML;
using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

public class TextColumnFinderTests
{
    public sealed class Pair
    {
        public string Source { get; set; } = "";
        public string First { get; set; } = "";
        public string Second { get; set; } = "";
        public float Score { get; set; }
    }

    private static readonly string[] Sources = ["airbnb-rtt", "airbnb-sampled", "policy-sampled", "news-rtt"];

    // A source tag before the two sentences — the shape of a sentence-similarity set.
    private static IDataView Pairs(int rows = 60) =>
        new MLContext(seed: 1).Data.LoadFromEnumerable(Enumerable.Range(0, rows).Select(i => new Pair
        {
            Source = Sources[i % Sources.Length],
            First = $"the room number {i} was clean and the host was kind",
            Second = $"a guest said room {i} felt tidy and welcoming overall",
            Score = i % 5
        }));

    [Fact]
    public void A_category_tag_is_not_taken_for_a_sentence()
    {
        Assert.Equal(["First", "Second"], TextColumnFinder.Find(Pairs(), "Score", 2));
    }

    [Fact]
    public void Columns_declared_text_come_first_in_declaration_order()
    {
        var overrides = new Dictionary<string, string> { ["Second"] = "text", ["First"] = "text" };

        Assert.Equal(["Second", "First"], TextColumnFinder.Find(Pairs(), "Score", 2, overrides));
    }

    [Fact]
    public void A_column_declared_anything_else_is_never_chosen()
    {
        var overrides = new Dictionary<string, string> { ["First"] = "categorical" };
        var log = new List<string>();

        var chosen = TextColumnFinder.Find(Pairs(), "Score", 2, overrides, log.Add);

        Assert.DoesNotContain("First", chosen);
        Assert.Equal("Second", chosen[0]);
    }

    [Fact]
    public void A_guess_is_reported_with_how_to_declare()
    {
        var log = new List<string>();

        // Only one column reads as text, so the second pick is the tag — said out loud.
        var chosen = TextColumnFinder.Find(Pairs(), "Second", 2, log: log.Add);

        Assert.Equal(["First", "Source"], chosen);
        Assert.Contains(log, l => l.Contains("Source") && l.Contains("type: text"));
    }

    [Fact]
    public void FindFirst_takes_the_text_over_the_tag()
    {
        Assert.Equal("First", TextColumnFinder.FindFirst(Pairs(), "Score"));
    }
}
