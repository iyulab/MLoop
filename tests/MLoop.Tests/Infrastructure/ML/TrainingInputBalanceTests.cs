using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.Infrastructure.ML;

/// <summary>
/// <c>--balance</c> / <c>POST /train</c> <c>balance</c> — one judgment of what can be balanced, and one way to balance:
/// the test part split off first, only the train part balanced.
/// </summary>
public sealed class TrainingInputBalanceTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "mloop-balance-" + Guid.NewGuid().ToString("N"));

    public TrainingInputBalanceTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private static ModelDefinition Definition(string task = "binary-classification", string? weight = null)
        => new() { Task = task, Label = "y", Weight = weight };

    [Theory]
    [InlineData(null)]
    [InlineData("none")]
    [InlineData("auto")]
    [InlineData("AUTO")]
    [InlineData("2.5")]
    public void AnOptionTheRunCanHonour_HasNoConflict(string? balance)
        => Assert.Null(TrainingInput.BalanceConflict(Definition(), balance));

    [Fact]
    public void ImageClassification_TakesABalance()
        => Assert.Null(TrainingInput.BalanceConflict(Definition("image-classification"), "3"));

    [Theory]
    [InlineData("binary-classification", null, "lots", "Invalid balance option")]
    [InlineData("binary-classification", null, "0", "Invalid balance option")]
    [InlineData("binary-classification", null, "2,5", "Invalid balance option")]   // a ratio is invariant: 2.5, not 2,5
    [InlineData("regression", null, "auto", "classification")]
    [InlineData("binary-classification", "w", "auto", "--balance")]
    [InlineData("object-detection", null, "3", "labels the boxes")]
    public void AnOptionTheRunCouldNotHonour_IsRefusedWithTheReason(string task, string? weight, string balance, string expected)
        => Assert.Contains(expected, TrainingInput.BalanceConflict(Definition(task, weight), balance));

    [Fact]
    public void WithATestSplit_OnlyTheTrainPartIsBalanced()
    {
        var file = Path.Combine(_dir, "data.csv");
        File.WriteAllLines(file, ["x,y", .. Enumerable.Range(0, 200).Select(i => $"{i},{(i < 20 ? "yes" : "no")}")]);

        var balanced = TrainingInput.Balance(file, "y", "2", testSplit: 0.2);

        Assert.True(balanced.Result.Applied);
        Assert.NotNull(balanced.TestFile);
        Assert.EndsWith("_balanced.csv", balanced.TrainFile);
        var test = File.ReadAllLines(balanced.TestFile!).Skip(1).ToList();
        Assert.Equal(4, test.Count(l => l.EndsWith(",yes")));            // the test part keeps its 1:9, stratified
        var train = File.ReadAllLines(balanced.TrainFile).Skip(1).ToList();
        Assert.Empty(train.Intersect(test));                             // no balanced copy of a test row
    }

    [Fact]
    public void WithoutATestSplit_TheWholeIsBalanced()
    {
        var file = Path.Combine(_dir, "whole.csv");
        File.WriteAllLines(file, ["x,y", .. Enumerable.Range(0, 100).Select(i => $"{i},{(i < 5 ? "yes" : "no")}")]);

        var balanced = TrainingInput.Balance(file, "y", "2", testSplit: 0);

        Assert.Null(balanced.TestFile);
        Assert.Null(balanced.Split);
        Assert.True(balanced.Result.Applied);
    }
}
