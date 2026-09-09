using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

/// <summary>
/// The two places a first-time user is told what to do — what <c>init</c> says the project looks
/// like, and what <c>train</c> says to do next — have to be true of the run the user just made.
/// </summary>
/// <remarks>
/// Both were found by walking the release procedure by hand rather than by any assertion: the
/// folder listing lined its comments up for one model name and no other, and the next steps asked
/// the reader to promote an experiment that the following lines reported as already promoted.
/// Neither changes an exit code, so nothing failed.
/// </remarks>
public class FirstSessionGuidanceTests
{
    [Theory]
    [InlineData("default")]
    [InlineData("churn")]
    [InlineData("a-considerably-longer-model-name")]
    public void TheFolderListingLinesUpForAnyModelName(string modelName)
    {
        var folders = InitCommand.FolderStructure(modelName, isObjectDetection: false, isDirectoryBased: false);

        var widths = folders.Select(f => f.Path.Length).Distinct().ToList();
        Assert.True(widths.Count == 1,
            $"comments start in different columns for '{modelName}': {string.Join(", ", widths)}");
        Assert.All(folders, f => Assert.EndsWith(" ", f.Path, StringComparison.Ordinal));
    }

    [Theory]
    [InlineData(true, false, "datasets/coco/")]
    [InlineData(false, true, "datasets/images/<class>/")]
    [InlineData(false, false, "datasets/")]
    public void TheDatasetRowMatchesTheTaskAndStillLinesUp(bool objectDetection, bool directoryBased, string expected)
    {
        var folders = InitCommand.FolderStructure("default", objectDetection, directoryBased);

        Assert.Equal(expected, folders[0].Path.TrimEnd());
        Assert.Single(folders.Select(f => f.Path.Length).Distinct());
    }

    [Fact]
    public void NextStepsDoNotAskForAPromotionThatIsAboutToHappen()
    {
        var steps = TrainPresenter.NextStepsAfterTraining("default", "exp-001", promotionFollows: true);

        Assert.DoesNotContain(steps, s => s.Contains("promote", StringComparison.Ordinal));
        Assert.Contains(steps, s => s.Contains("predict", StringComparison.Ordinal));
    }

    [Fact]
    public void NextStepsOfferPromotionWhenTheCommandWillNotDoIt()
    {
        // mloop train --no-promote: the experiment stays in staging, so promoting it is genuinely
        // the next thing to do and naming the experiment is what makes the line runnable.
        var steps = TrainPresenter.NextStepsAfterTraining("churn", "exp-007", promotionFollows: false);

        Assert.Contains(steps, s => s == "mloop promote exp-007 --name churn");
    }
}
