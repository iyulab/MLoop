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

    [Theory]
    [InlineData("mean")]
    [InlineData("avg")]
    [InlineData("median")]
    [InlineData("variance")]
    [InlineData("first")]
    [InlineData("LAST")]
    [InlineData("stdev")]
    public void Rolling_and_resample_accept_every_aggregation_FilePrepper_can_compute(string method)
    {
        Assert.Empty(PrepStepRules.Problems(new PrepStep { Type = "rolling", WindowSize = 3, Columns = ["x"], Method = method }));
        Assert.Empty(PrepStepRules.Problems(new PrepStep { Type = "resample", TimeColumn = "t", Window = "1H", Columns = ["x"], Method = method }));
    }

    [Fact]
    public void An_aggregation_the_executor_cannot_run_is_refused_before_it_runs()
    {
        var problems = PrepStepRules.Problems(new PrepStep { Type = "rolling", WindowSize = 3, Columns = ["x"], Method = "mode" }).ToList();

        Assert.Single(problems);
        Assert.Contains("'rolling' method 'mode' is not one of: mean, min, max, sum, count, std, var, median, first, last", problems[0]);
    }

    [Fact]
    public void Every_listed_aggregation_parses()
    {
        foreach (var name in PrepStepRules.AggregationMethods)
            Assert.True(PrepStepRules.TryParseAggregation(name, out _), name);
    }

    [Theory]
    [InlineData("gt", "abc", true)]
    [InlineData(">=", "", true)]
    [InlineData("gt", "10", false)]
    [InlineData("eq", "abc", false)]
    [InlineData("contains", "abc", false)]
    public void A_number_operator_needs_a_number_for_a_value(string op, string value, bool refused)
    {
        var problems = PrepStepRules.Problems(new PrepStep { Type = "filter-rows", Column = "x", Operator = op, Value = value }).ToList();

        Assert.Equal(refused, problems.Count > 0);
    }

    [Fact]
    public void An_unknown_filter_operator_is_refused_by_name()
    {
        var problems = PrepStepRules.Problems(new PrepStep { Type = "filter-rows", Column = "x", Operator = "startswith", Value = "a" }).ToList();

        Assert.Contains(problems, p => p.Contains("'filter-rows' operator 'startswith' is not one of", StringComparison.Ordinal));
    }

    [Fact]
    public async Task A_filter_value_that_is_not_a_number_names_the_step_instead_of_a_format_exception()
    {
        var input = Path.Combine(_dir, "f.csv");
        await File.WriteAllTextAsync(input, "x\n1\n2\n");

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => new DataPipelineExecutor(new TestLogger())
            .ExecuteAsync(input, [new PrepStep { Type = "filter-rows", Column = "x", Operator = "gt", Value = "abc" }], Path.Combine(_dir, "o.csv")));

        Assert.Contains("filter-rows operator 'gt' compares numbers, but value 'abc' is not a number", ex.Message, StringComparison.Ordinal);
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
