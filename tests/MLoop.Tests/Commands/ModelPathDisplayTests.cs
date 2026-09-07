using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

/// <summary>
/// Where a trained model landed is shown as the project reads it, not as the filesystem spells it.
/// </summary>
/// <remarks>
/// The absolute path is long enough that a redirected stdout — 80 columns, which is what a script or
/// a CI log gets — folds it mid-token and splits <c>model.zip</c> across two lines. A user copying
/// that out has a broken path, and nothing says so. From the models directory down is the part that
/// identifies the artifact and the vocabulary every experiment-taking command already uses.
/// </remarks>
public class ModelPathDisplayTests
{
    [Theory]
    [InlineData("models", "default", "staging", "exp-001")]
    [InlineData("models", "churn", "production", "")]
    public void TheProjectRelativePartIsWhatIsShown(string models, string name, string stage, string experiment)
    {
        var segments = new[] { "C:", "a-long-user-directory", "deeper", "project", models, name, stage }
            .Concat(experiment.Length == 0 ? [] : new[] { experiment })
            .Append("model.zip");
        var absolute = string.Join(Path.DirectorySeparatorChar, segments);

        var shown = TrainPresenter.WhereItLandedInTheProject(absolute);

        Assert.StartsWith(models + Path.DirectorySeparatorChar, shown, StringComparison.Ordinal);
        Assert.EndsWith("model.zip", shown, StringComparison.Ordinal);
        Assert.True(shown.Length < absolute.Length, "the shown path is not shorter than the absolute one");
        // 80 columns is what Spectre assumes when stdout is redirected; the label costs 16 of them.
        Assert.True(shown.Length <= 64, $"'{shown}' still folds at 80 columns beside its label");
    }

    [Fact]
    public void APathOutsideTheProjectLayoutIsLeftWhole()
    {
        // Guessing at a path this method does not recognise would be worse than showing all of it.
        var elsewhere = string.Join(Path.DirectorySeparatorChar, "C:", "somewhere", "else", "model.zip");

        Assert.Equal(elsewhere, TrainPresenter.WhereItLandedInTheProject(elsewhere));
    }
}
