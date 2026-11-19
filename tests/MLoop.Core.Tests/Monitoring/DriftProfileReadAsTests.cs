using MLoop.Core.Monitoring;

namespace MLoop.Core.Tests.Monitoring;

/// <summary>
/// A text task reads its text columns and nothing else, so a profile of its training table keeps only those. A question-answering
/// table carries the answer's position as a number; profiled as an input it read as the largest drift there is on every window,
/// because serving traffic (a passage and a question) never has it.
/// </summary>
public class DriftProfileReadAsTests
{
    private static DriftProfile Profile() =>
        DriftProfile.Build(Enumerable.Range(0, 200).Select(i => new[] { (i % 50).ToString(System.Globalization.CultureInfo.InvariantCulture) }).ToArray(),
            [("answer_start", true)]);

    [Theory]
    [InlineData("question-answering")]
    [InlineData("ner")]
    [InlineData("sentence-similarity")]
    [InlineData("text-classification")]
    public void ATextTaskReadsNoNumericColumnOfItsTable(string task) =>
        Assert.Empty(Profile().ReadAs(task).Features);

    [Theory]
    [InlineData("binary-classification")]
    [InlineData("regression")]
    [InlineData(null)]
    public void AnyOtherTaskKeepsItsProfileAsItWas(string? task)
    {
        var profile = Profile();
        Assert.Same(profile, profile.ReadAs(task));
        Assert.Single(profile.ReadAs(task).Features);
    }

    [Fact]
    public void ATextTasksTextProfilesAreKept()
    {
        var withText = Profile() with { Texts = [TextProfile.Build("context", ["a b c", "d e f", "a d g"])] };

        var read = withText.ReadAs("question-answering");

        Assert.Empty(read.Features);
        Assert.Single(read.Texts!);
    }
}
