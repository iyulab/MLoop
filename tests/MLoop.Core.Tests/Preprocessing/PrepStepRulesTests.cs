using MLoop.Core.Preprocessing;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Core.Tests.Preprocessing;

/// <summary>
/// One vocabulary of prep steps: the executor runs every step type the validators accept, and a
/// step the executor does not know is refused by the same message everywhere.
/// </summary>
public class PrepStepRulesTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"mloop-prep-rules-{Guid.NewGuid():N}");

    public PrepStepRulesTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }

    [Theory]
    [InlineData("fill_missing", "fill-missing")]
    [InlineData("Scale", "normalize")]
    [InlineData("data_sampling", "sample")]
    [InlineData("SPLIT_LIST", "split-list")]
    [InlineData("explode", "explode")]
    public void Spellings_and_aliases_resolve_to_one_name(string type, string canonical)
        => Assert.Equal(canonical, PrepStepRules.Canonical(type));

    [Theory]
    [InlineData("")]
    [InlineData("unpivot")]
    [InlineData("explodes")]
    public void An_unknown_type_has_no_canonical_name(string type)
        => Assert.Null(PrepStepRules.Canonical(type));

    [Fact]
    public async Task The_executor_dispatches_every_step_type()
    {
        // Every canonical name reaches its own handler: a bare step may fail on its missing
        // parameters, but never as an unknown type.
        var input = Path.Combine(_dir, "in.csv");
        await File.WriteAllTextAsync(input, "a,b\n1,2\n");
        var executor = new DataPipelineExecutor(new TestLogger());

        var unknown = new List<string>();
        foreach (var name in PrepStepRules.CanonicalNames)
        {
            try
            {
                await executor.ExecuteAsync(input, [new PrepStep { Type = name }], Path.Combine(_dir, $"{name}.csv"));
            }
            catch (Exception ex) when (ex.Message.Contains("Unknown prep step type"))
            {
                unknown.Add(name);
            }
            catch
            {
                // A handler ran and refused the bare step — dispatch worked.
            }
        }

        Assert.Empty(unknown);
    }

    [Fact]
    public async Task The_dispatch_check_sees_an_unknown_type()
    {
        var input = Path.Combine(_dir, "in.csv");
        await File.WriteAllTextAsync(input, "a\n1\n");

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => new DataPipelineExecutor(new TestLogger())
            .ExecuteAsync(input, [new PrepStep { Type = "unpivot" }], Path.Combine(_dir, "out.csv")));

        Assert.Contains("Unknown prep step type 'unpivot'", ex.Message);
    }

    [Fact]
    public void List_steps_need_their_columns()
    {
        Assert.Contains("'explode' requires 'columns' or 'column'", PrepStepRules.Problems(new PrepStep { Type = "explode" }));
        Assert.Empty(PrepStepRules.Problems(new PrepStep { Type = "split-list", Columns = ["f"] }));
    }

    [Fact]
    public async Task Explode_then_split_list_turn_a_group_row_into_rows_with_feature_columns()
    {
        var input = Path.Combine(_dir, "q.csv");
        await File.WriteAllTextAsync(input, "query,label,features\n7,\"[2,0]\",\"[[0.5,1],[0.25,0]]\"\n");
        var output = Path.Combine(_dir, "docs.csv");

        await new DataPipelineExecutor(new TestLogger()).ExecuteAsync(input,
        [
            new PrepStep { Type = "explode", Columns = ["label", "features"] },
            new PrepStep { Type = "split-list", Column = "features" },
        ], output);

        Assert.Equal(
            ["query,label,features.0,features.1", "7,2,0.5,1", "7,0,0.25,0"],
            (await File.ReadAllLinesAsync(output)).Select(l => l.TrimStart('﻿')));
    }

    private sealed class TestLogger : ILogger
    {
        public void Debug(string message) { }
        public void Info(string message) { }
        public void Warning(string message) { }
        public void Error(string message) { }
        public void Error(string message, Exception exception) { }
    }
}
