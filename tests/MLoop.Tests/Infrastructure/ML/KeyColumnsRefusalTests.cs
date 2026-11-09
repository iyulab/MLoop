using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.Infrastructure.ML;

/// <summary>
/// A ranking or recommendation project names the columns it reads as the query group, the user and the item, and
/// <c>mloop init</c> writes placeholders for them (<c>query_id</c>, <c>user_id</c>, <c>item_id</c>). A table of the user's own rarely
/// spells them that way: the refusal says which file lacks which column, lists the columns it has, and names the option that fixes it —
/// before any data is read, not as "Group column 'query_id' not found in data." from the trainer.
/// </summary>
public sealed class KeyColumnsRefusalTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "mloop-keycols-" + Guid.NewGuid().ToString("N"));

    public KeyColumnsRefusalTests() => Directory.CreateDirectory(_directory);

    public void Dispose()
    {
        try { Directory.Delete(_directory, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private string Write(string name, string header)
    {
        var path = Path.Combine(_directory, name);
        File.WriteAllText(path, header + "\n1,2,3\n");
        return path;
    }

    [Fact]
    public void A_ranking_project_whose_group_column_is_not_in_the_file_names_the_file_the_column_the_columns_and_the_option()
    {
        var path = Write("rank.csv", "qid,f1,f2,label");
        var refusal = TrainingInput.KeyColumnsRefusal(path, new ModelDefinition { Label = "label", Task = "ranking", GroupColumn = "query_id" });

        Assert.NotNull(refusal);
        Assert.Contains("rank.csv", refusal.Message, StringComparison.Ordinal);
        Assert.Contains("'query_id'", refusal.Message, StringComparison.Ordinal);
        Assert.Contains("Available columns: qid, f1, f2, label", refusal.Message, StringComparison.Ordinal);
        Assert.Contains("--group-column <column>", refusal.Hint, StringComparison.Ordinal);
        Assert.Contains("group_column", refusal.Hint, StringComparison.Ordinal);
    }

    [Fact]
    public void A_recommendation_project_missing_both_columns_names_both_options()
    {
        var path = Write("ratings.csv", "user,item,rating");
        var refusal = TrainingInput.KeyColumnsRefusal(path, new ModelDefinition { Label = "label", Task = "recommendation", UserColumn = "user_id", ItemColumn = "item_id" });

        Assert.NotNull(refusal);
        Assert.Contains("'user_id'", refusal.Message, StringComparison.Ordinal);
        Assert.Contains("'item_id'", refusal.Message, StringComparison.Ordinal);
        Assert.Contains("--user-column <column>", refusal.Hint, StringComparison.Ordinal);
        Assert.Contains("--item-column <column>", refusal.Hint, StringComparison.Ordinal);
    }

    [Fact]
    public void A_recommendation_project_missing_one_column_names_only_that_one()
    {
        var path = Write("ratings.csv", "user,rating");
        var refusal = TrainingInput.KeyColumnsRefusal(path, new ModelDefinition { Label = "label", Task = "recommendation", UserColumn = "user", ItemColumn = "item" });

        Assert.NotNull(refusal);
        Assert.Contains("the item column 'item'", refusal.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("user column", refusal.Message, StringComparison.Ordinal);
        Assert.Contains("--item-column <column>", refusal.Hint, StringComparison.Ordinal);
        Assert.DoesNotContain("--user-column", refusal.Hint, StringComparison.Ordinal);
    }

    [Fact]
    public void Columns_the_file_has_and_tasks_without_key_columns_are_not_refused()
    {
        var path = Write("ok.csv", "qid,user,item,f1,label");
        Assert.Null(TrainingInput.KeyColumnsRefusal(path, new ModelDefinition { Label = "label", Task = "ranking", GroupColumn = "qid" }));
        Assert.Null(TrainingInput.KeyColumnsRefusal(path, new ModelDefinition { Label = "label", Task = "recommendation", UserColumn = "user", ItemColumn = "item" }));
        Assert.Null(TrainingInput.KeyColumnsRefusal(path, new ModelDefinition { Label = "label", Task = "regression", GroupColumn = "query_id" }));
    }

    [Fact]
    public void A_file_with_no_readable_header_is_left_to_the_other_refusals()
    {
        var path = Path.Combine(_directory, "empty.csv");
        File.WriteAllText(path, "");
        Assert.Null(TrainingInput.KeyColumnsRefusal(path, new ModelDefinition { Label = "label", Task = "ranking", GroupColumn = "query_id" }));
    }
}
