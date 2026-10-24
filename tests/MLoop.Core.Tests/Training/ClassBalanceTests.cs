using MLoop.Core.Training;

namespace MLoop.Core.Tests.Training;

public class ClassBalanceTests
{
    private static List<KeyValuePair<string, IReadOnlyList<int>>> Classes(params (string Name, int Count)[] classes)
    {
        var next = 0;
        return classes
            .Select(c => KeyValuePair.Create(c.Name, (IReadOnlyList<int>)Enumerable.Range(next, c.Count).ToList()))
            .Select(c => { next += c.Value.Count; return c; })
            .ToList();
    }

    [Fact]
    public void Oversample_ARatio_RaisesEverySmallerClassToTheLargestOverTheRatio()
    {
        // The shape of a defect set: 602 good, 60 defective, --balance 3.
        var classes = Classes(("OK", 602), ("NG", 60));

        var outcome = ClassBalance.Oversample(classes, "3");

        Assert.True(outcome.Applied);
        Assert.Equal(201 - 60, outcome.Copies.Count); // ⌈602 ÷ 3⌉ = 201
        Assert.All(outcome.Copies, copy => Assert.Contains(copy, classes[1].Value));
        Assert.Equal(60, outcome.OriginalMinorityCount);
        Assert.Equal(201, outcome.NewMinorityCount);
        Assert.Equal(602 / 201.0, outcome.NewRatio, 9);
    }

    [Fact]
    public void Oversample_DrawsTheSameCopiesEveryRun()
    {
        var classes = Classes(("a", 50), ("b", 5), ("c", 3));

        var first = ClassBalance.Oversample(classes, "2");
        var second = ClassBalance.Oversample(classes, "2");

        Assert.Equal(first.Copies, second.Copies);
        Assert.Equal((25 - 5) + (25 - 3), first.Copies.Count);
    }

    [Fact]
    public void Oversample_Auto_LeavesARatioAtTheThresholdAlone()
    {
        var outcome = ClassBalance.Oversample(Classes(("a", 100), ("b", 10)), "auto");

        Assert.False(outcome.Applied);
        Assert.Contains("within acceptable range", outcome.Message);
    }

    [Fact]
    public void Oversample_Auto_BalancesAboveTheThresholdToTheDefaultRatio()
    {
        var outcome = ClassBalance.Oversample(Classes(("a", 200), ("b", 5)), "auto");

        Assert.Equal(20 - 5, outcome.Copies.Count); // 200 ÷ 10
    }

    [Fact]
    public void Oversample_OneClass_DoesNothing()
    {
        Assert.False(ClassBalance.Oversample(Classes(("a", 10)), "2").Applied);
    }

    [Theory]
    [InlineData(null, false)]
    [InlineData("", false)]
    [InlineData("none", false)]
    [InlineData("NONE", false)]
    [InlineData("auto", true)]
    [InlineData("3", true)]
    public void Requested_IsTrueForAnythingButNone(string? option, bool expected)
    {
        Assert.Equal(expected, ClassBalance.Requested(option));
    }

    [Theory]
    [InlineData("auto")]
    [InlineData("2.5")]
    [InlineData(null)]
    public void OptionError_AcceptsTheBalanceOptions(string? option)
    {
        Assert.Null(ClassBalance.OptionError(option));
    }

    [Theory]
    [InlineData("-1")]
    [InlineData("0")]
    [InlineData("smote")]
    public void OptionError_RefusesAnythingElse(string option)
    {
        Assert.NotNull(ClassBalance.OptionError(option));
    }
}
