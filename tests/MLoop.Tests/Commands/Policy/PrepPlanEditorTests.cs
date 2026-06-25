using MLoop.CLI.Commands.Policy;
using MLoop.Core.Preprocessing;

namespace MLoop.Tests.Commands.Policy;

public class PrepPlanEditorTests
{
    [Fact]
    public void SetStep_NewStep_Appends()
    {
        var steps = new List<PrepStep>();
        PrepPlanEditor.SetStep(steps, new PrepStep { Type = "normalize", Method = "z-score", Columns = ["a"] });
        Assert.Single(steps);
        Assert.Equal("normalize", steps[0].Type);
    }

    [Fact]
    public void SetStep_SameTypeAndColumns_ReplacesNotDuplicates()
    {
        var steps = new List<PrepStep> { new() { Type = "normalize", Method = "min-max", Columns = ["a", "b"] } };
        PrepPlanEditor.SetStep(steps, new PrepStep { Type = "normalize", Method = "z-score", Columns = ["b", "a"] }); // 순서 무관
        Assert.Single(steps);
        Assert.Equal("z-score", steps[0].Method); // 덮어씀
    }

    [Fact]
    public void SetStep_SameTypeDifferentColumns_AddsSecond()
    {
        var steps = new List<PrepStep> { new() { Type = "normalize", Columns = ["a"] } };
        PrepPlanEditor.SetStep(steps, new PrepStep { Type = "normalize", Columns = ["b"] });
        Assert.Equal(2, steps.Count); // 다른 컬럼군 → 별도 step
    }

    [Fact]
    public void RemoveStep_MatchingType_RemovesAndReturnsCount()
    {
        var steps = new List<PrepStep> { new() { Type = "normalize", Columns = ["a"] }, new() { Type = "scale", Columns = ["b"] } };
        var removed = PrepPlanEditor.RemoveStep(steps, "normalize", columns: null);
        Assert.Equal(1, removed);
        Assert.Single(steps);
        Assert.Equal("scale", steps[0].Type);
    }

    [Fact]
    public void RemoveStep_MatchingTypeAndColumns_RemovesOnlyExactColumnMatch()
    {
        var steps = new List<PrepStep>
        {
            new() { Type = "normalize", Columns = ["a"] },
            new() { Type = "normalize", Columns = ["b"] }
        };
        var removed = PrepPlanEditor.RemoveStep(steps, "normalize", columns: ["a"]);
        Assert.Equal(1, removed);
        Assert.Single(steps);
        Assert.Equal("b", steps[0].Columns![0]); // only the ["a"] step removed
    }

    [Theory]
    [InlineData("normalize:z-score", "normalize", "z-score")]
    [InlineData("drop-duplicates", "drop-duplicates", null)]
    public void ParseSet_SplitsTypeAndMethod(string arg, string type, string? method)
    {
        var (t, m) = PrepPlanEditor.ParseSet(arg);
        Assert.Equal(type, t);
        Assert.Equal(method, m);
    }
}
