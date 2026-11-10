using MLoop.Core.Storage;

namespace MLoop.Core.Tests.Storage;

public class ProjectLayoutTests
{
    [Fact]
    public void PredictionFileName_IsModelThenKindThenStamp()
    {
        var at = new DateTimeOffset(2026, 3, 4, 15, 6, 7, TimeSpan.Zero).ToLocalTime();

        Assert.Equal(
            $"churn-predictions-{at:yyyyMMdd-HHmmss}.csv",
            ProjectLayout.PredictionFileName("churn", ProjectLayout.PredictionKind.Rows, at));
        Assert.Equal(
            $"churn-detections-{at:yyyyMMdd-HHmmss}.json",
            ProjectLayout.PredictionFileName("churn", ProjectLayout.PredictionKind.Detections, at));
        Assert.Equal(
            $"churn-forecast-{at:yyyyMMdd-HHmmss}.csv",
            ProjectLayout.PredictionFileName("churn", ProjectLayout.PredictionKind.Forecast, at));
    }

    /// <summary>
    /// The name a person scans a directory listing for is in their own zone — the same rule the
    /// display authority states for file names, and the opposite of the one for storage keys.
    /// </summary>
    [Fact]
    public void PredictionFileName_StampsTheReadersClock()
    {
        var utcNoon = new DateTimeOffset(2026, 3, 4, 12, 0, 0, TimeSpan.Zero);

        var name = ProjectLayout.PredictionFileName("m", ProjectLayout.PredictionKind.Rows, utcNoon);

        Assert.Contains(utcNoon.ToLocalTime().ToString("yyyyMMdd-HHmmss"), name);
    }

    /// <summary>
    /// Two runs of a script in one second were given one name and the second replaced the first: a 700-row output was
    /// silently replaced by a 60-row one. The path asked for is never one that already exists.
    /// </summary>
    [Fact]
    public void AvailablePredictionPath_IsTheConventionalNameWhenFree_AndNeverAnExistingFile()
    {
        var dir = Directory.CreateTempSubdirectory("layout-").FullName;
        try
        {
            var at = new DateTimeOffset(2026, 3, 4, 15, 6, 7, TimeSpan.Zero);
            var name = ProjectLayout.PredictionFileName("churn", ProjectLayout.PredictionKind.Rows, at);

            var first = ProjectLayout.AvailablePredictionPath(dir, "churn", ProjectLayout.PredictionKind.Rows, at);
            Assert.Equal(Path.Combine(dir, name), first);

            File.WriteAllText(first, "first run");
            var second = ProjectLayout.AvailablePredictionPath(dir, "churn", ProjectLayout.PredictionKind.Rows, at);
            File.WriteAllText(second, "second run");
            var third = ProjectLayout.AvailablePredictionPath(dir, "churn", ProjectLayout.PredictionKind.Rows, at);

            Assert.Equal(Path.Combine(dir, Path.GetFileNameWithoutExtension(name) + "-2.csv"), second);
            Assert.Equal(Path.Combine(dir, Path.GetFileNameWithoutExtension(name) + "-3.csv"), third);
            Assert.Equal("first run", File.ReadAllText(first));
            Assert.Equal("second run", File.ReadAllText(second));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// The file is written when the prediction ends, so a second run that asks for a path while the first is still predicting
    /// must not be given the first one's: the name is taken when it is chosen, not when it is written. (Parallel runs of one
    /// script in one second chose one name and the later writer replaced the earlier.)
    /// </summary>
    [Fact]
    public void AvailablePredictionPath_ReservesTheNameWhenItIsChosen_BeforeAnythingIsWritten()
    {
        var dir = Directory.CreateTempSubdirectory("layout-").FullName;
        try
        {
            var at = new DateTimeOffset(2026, 3, 4, 15, 6, 7, TimeSpan.Zero);

            var first = ProjectLayout.AvailablePredictionPath(dir, "churn", ProjectLayout.PredictionKind.Rows, at);
            var second = ProjectLayout.AvailablePredictionPath(dir, "churn", ProjectLayout.PredictionKind.Rows, at);   // the first is still "predicting"

            Assert.NotEqual(first, second);
            Assert.True(File.Exists(first));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void AvailablePredictionPath_GivesParallelAskersDistinctNames()
    {
        var dir = Directory.CreateTempSubdirectory("layout-").FullName;
        try
        {
            var at = new DateTimeOffset(2026, 3, 4, 15, 6, 7, TimeSpan.Zero);

            var paths = Enumerable.Range(0, 32).AsParallel().WithDegreeOfParallelism(16)
                .Select(_ => ProjectLayout.AvailablePredictionPath(dir, "churn", ProjectLayout.PredictionKind.Rows, at)).ToList();

            Assert.Equal(32, paths.Distinct().Count());
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>The reader of prediction files finds the suffixed ones too: the search patterns end in a wildcard before the extension.</summary>
    [Fact]
    public void AvailablePredictionPath_SuffixedNamesAreFoundByTheSearchPatterns()
    {
        var dir = Directory.CreateTempSubdirectory("layout-").FullName;
        try
        {
            var at = new DateTimeOffset(2026, 3, 4, 15, 6, 7, TimeSpan.Zero);
            foreach (var kind in Enum.GetValues<ProjectLayout.PredictionKind>())
            {
                var first = ProjectLayout.AvailablePredictionPath(dir, "churn", kind, at);
                File.WriteAllText(first, "x");
                File.WriteAllText(ProjectLayout.AvailablePredictionPath(dir, "churn", kind, at), "y");
            }

            var found = ProjectLayout.PredictionSearchPatterns("churn")
                .SelectMany(pattern => Directory.EnumerateFiles(dir, pattern))
                .Count();

            Assert.Equal(2 * Enum.GetValues<ProjectLayout.PredictionKind>().Length, found);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void PredictionFileName_NormalizesTheModelName()
        => Assert.StartsWith(
            "churn-",
            ProjectLayout.PredictionFileName("  Churn  ", ProjectLayout.PredictionKind.Rows, DateTimeOffset.Now));

    /// <summary>
    /// The defect this authority exists for: `status` looked for the tabular convention only, so a
    /// model that writes detections or a forecast reported "no predictions" forever. Every kind a
    /// writer can produce has to be findable.
    /// </summary>
    [Fact]
    public void EveryKindAWriterProduces_IsFoundByAPattern()
    {
        var at = DateTimeOffset.Now;
        var patterns = ProjectLayout.PredictionSearchPatterns("churn");

        foreach (var kind in ProjectLayout.PredictionKinds)
        {
            var name = ProjectLayout.PredictionFileName("churn", kind, at);
            Assert.True(
                patterns.Any(p => Matches(p, name)),
                $"No search pattern finds a {kind} file ('{name}'). Patterns: {string.Join(", ", patterns)}");
        }
    }

    /// <summary>
    /// One pattern per kind rather than a single <c>{model}-*</c>: model names may contain hyphens,
    /// and a blanket glob for <c>my</c> would swallow <c>my-other</c>'s output.
    /// </summary>
    [Fact]
    public void APatternDoesNotMatchAnotherModelWhoseNameExtendsThisOne()
    {
        var at = DateTimeOffset.Now;
        var patterns = ProjectLayout.PredictionSearchPatterns("my");
        var otherModelsFile = ProjectLayout.PredictionFileName("my-other", ProjectLayout.PredictionKind.Rows, at);

        Assert.DoesNotContain(patterns, p => Matches(p, otherModelsFile));
    }

    [Fact]
    public void PatternsCoverExactlyTheKinds()
        => Assert.Equal(ProjectLayout.PredictionKinds.Count(), ProjectLayout.PredictionSearchPatterns("m").Count);

    /// <summary>
    /// `datasets/` names are shared with the API, which cannot reach the CLI class that used to
    /// hold them privately.
    /// </summary>
    [Fact]
    public void DatasetNamesAreTheConventionTheDocsDescribe()
    {
        Assert.Equal("datasets", ProjectLayout.DatasetsDirectory);
        Assert.Equal("predictions", ProjectLayout.PredictionsDirectory);
        Assert.Equal("train.csv", ProjectLayout.TrainFileName);
        Assert.Equal("test.csv", ProjectLayout.TestFileName);
        Assert.Equal("predict.csv", ProjectLayout.PredictFileName);
        Assert.Equal("validation.csv", ProjectLayout.ValidationFileName);
    }

    /// <summary>
    /// A project-root name is not a model-directory name: the two authorities must not claim the
    /// same string, or a reader cannot tell which contract a literal came from.
    /// </summary>
    [Fact]
    public void DoesNotCollideWithTheModelDirectoryLayout()
    {
        Assert.NotEqual(ExperimentLayout.ModelsDirectory, ProjectLayout.DatasetsDirectory);
        Assert.NotEqual(ExperimentLayout.ModelsDirectory, ProjectLayout.PredictionsDirectory);
    }

    /// <summary>Minimal glob matcher — the patterns use one trailing <c>*</c> only.</summary>
    private static bool Matches(string pattern, string name)
    {
        var parts = pattern.Split('*');
        Assert.Equal(2, parts.Length);
        return name.StartsWith(parts[0], StringComparison.Ordinal)
            && name.EndsWith(parts[1], StringComparison.Ordinal)
            && name.Length >= parts[0].Length + parts[1].Length;
    }
}
