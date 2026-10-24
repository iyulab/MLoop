using MLoop.Core.Storage;

namespace MLoop.Core.Tests.Storage;

public class ProjectPathsTests
{
    private static readonly string Root = Path.Combine(Path.GetTempPath(), "proj");

    [Fact]
    public void ARelativePath_IsTheProjects()
    {
        Assert.Equal(Path.Combine(Root, "datasets", "train.csv"), ProjectPaths.Resolve(Path.Combine("datasets", "train.csv"), Root));
    }

    [Fact]
    public void ARootedPath_IsItself_Normalized()
    {
        var rooted = Path.Combine(Path.GetTempPath(), "elsewhere", "..", "x.csv");

        Assert.Equal(Path.Combine(Path.GetTempPath(), "x.csv"), ProjectPaths.Resolve(rooted, Root));
    }

    [Fact]
    public void WithoutAProject_ARelativePath_IsTheWorkingDirectorys()
    {
        Assert.Equal(Path.Combine(Directory.GetCurrentDirectory(), "x.csv"), ProjectPaths.Resolve("x.csv", projectRoot: null));
    }

    [Theory]
    [InlineData("datasets/train.csv", true)]
    [InlineData(".", true)]
    [InlineData("a/../b.csv", true)]
    [InlineData("../proj-other/train.csv", false)]   // shares the root's name as a prefix
    [InlineData("../train.csv", false)]
    [InlineData("..", false)]
    [InlineData("..data/x.csv", true)]                // a directory whose name begins with two dots
    public void IsWithin_ComparesSegments_NotAPrefix(string relative, bool within)
    {
        Assert.Equal(within, ProjectPaths.IsWithin(ProjectPaths.Resolve(relative, Root), Root));
    }

    [Fact]
    public void IsWithin_RefusesAnotherRoot()
    {
        var other = OperatingSystem.IsWindows() ? @"Z:\data.csv" : "/data.csv";

        Assert.False(ProjectPaths.IsWithin(other, Root));
    }
}
