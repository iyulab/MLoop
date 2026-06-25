using MLoop.CLI.Commands.Policy;
using MLoop.CLI.Infrastructure.Configuration;

namespace MLoop.Tests.Commands.Policy;

public class FeatureSelectorTests
{
    [Fact]
    public void Drop_SetsIgnoreOverride()
    {
        var cols = new Dictionary<string, ColumnOverride>();
        FeatureSelector.Drop(cols, ["id", "ts"]);
        Assert.Equal("ignore", cols["id"].Type);
        Assert.Equal("ignore", cols["ts"].Type);
    }

    [Fact]
    public void KeepComplement_IgnoresEverythingExceptKeepAndLabel()
    {
        var all = new[] { "a", "b", "c", "y" };
        var ignore = FeatureSelector.KeepComplement(all, keep: ["a"], label: "y");
        Assert.Equal(new[] { "b", "c" }, ignore); // a(keep)·y(label) 제외
    }

    [Fact]
    public void KeepComplement_LabelAlwaysKept_CaseInsensitive()
    {
        var all = new[] { "A", "B", "Y" };
        var ignore = FeatureSelector.KeepComplement(all, keep: ["a"], label: "y");
        Assert.DoesNotContain("A", ignore); // keep a == A
        Assert.DoesNotContain("Y", ignore); // label y == Y
        Assert.Contains("B", ignore);
    }

    [Fact]
    public void Reset_RemovesOnlyIgnoreOverrides()
    {
        var cols = new Dictionary<string, ColumnOverride>
        {
            ["a"] = new() { Type = "ignore" },
            ["b"] = new() { Type = "categorical" }
        };
        var removed = FeatureSelector.Reset(cols);
        Assert.Equal(1, removed);
        Assert.False(cols.ContainsKey("a"));
        Assert.True(cols.ContainsKey("b")); // 타입 힌트 보존
    }

    [Fact]
    public void ApplyKeep_IgnoresComplement_PreservesExistingDescription()
    {
        var cols = new Dictionary<string, ColumnOverride>
        {
            ["b"] = new() { Type = "categorical", Description = "keep-this-note" }
        };
        FeatureSelector.ApplyKeep(cols, allColumns: ["a", "b", "c", "y"], keep: ["a"], label: "y");
        Assert.Equal("ignore", cols["b"].Type);
        Assert.Equal("keep-this-note", cols["b"].Description); // Description carried forward
        Assert.Equal("ignore", cols["c"].Type);
        Assert.False(cols.ContainsKey("a")); // kept → not ignored
        Assert.False(cols.ContainsKey("y")); // label → not ignored
    }
}
