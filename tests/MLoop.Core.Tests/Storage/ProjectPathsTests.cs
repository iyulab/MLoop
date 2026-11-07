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

    [Fact]
    public void IsSame_ComparesPlaces_NotSpellings()
    {
        Assert.True(ProjectPaths.IsSame(Path.Combine(Root, "a", "..", "b"), Path.Combine(Root, "b")));
        Assert.False(ProjectPaths.IsSame(Path.Combine(Root, "b"), Path.Combine(Root, "b2")));
        Assert.False(ProjectPaths.IsSame(Path.Combine(Root, "b"), Path.Combine(Root, "b", "c")));
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

/// <summary>
/// <see cref="ProjectPaths.IsWithin"/> compares the spelling of a path, so a symbolic link in the project that points out of it
/// stayed "within" — <c>POST /train</c> read the file the link named. <see cref="ProjectPaths.IsWithinReal"/> follows the links.
/// </summary>
public class ProjectPathsSymlinkTests : IDisposable
{
    private readonly string _base = Path.Combine(Path.GetTempPath(), $"mloop-links-{Guid.NewGuid():N}");
    private string Project => Path.Combine(_base, "proj");
    private string Outside => Path.Combine(_base, "outside");

    public ProjectPathsSymlinkTests()
    {
        Directory.CreateDirectory(Path.Combine(Project, "datasets"));
        Directory.CreateDirectory(Outside);
        File.WriteAllText(Path.Combine(Outside, "secret.csv"), "a,b\n1,2\n");
        File.WriteAllText(Path.Combine(Project, "datasets", "real.csv"), "a,b\n1,2\n");
    }

    public void Dispose()
    {
        try { Directory.Delete(_base, recursive: true); } catch (IOException) { }
        GC.SuppressFinalize(this);
    }

    private void Link(string link, string target, bool directory)
    {
        try
        {
            if (directory)
                Directory.CreateSymbolicLink(link, target);
            else
                File.CreateSymbolicLink(link, target);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or PlatformNotSupportedException)
        {
            Assert.Skip($"this machine cannot create symbolic links here: {ex.Message}");
        }
    }

    [Fact]
    public void ALinkedFile_ThatLeavesTheProject_IsOutside_ThoughItsSpellingIsInside()
    {
        var link = Path.Combine(Project, "datasets", "link.csv");
        Link(link, Path.Combine(Outside, "secret.csv"), directory: false);

        Assert.True(ProjectPaths.IsWithin(link, Project));            // the spelling: the lexical check cannot see it
        Assert.False(ProjectPaths.IsWithinReal(link, Project));       // the place it names
    }

    [Fact]
    public void ALinkedDirectory_ThatLeavesTheProject_IsOutside_ForEveryFileUnderIt()
    {
        var link = Path.Combine(Project, "datasets", "shared");
        Link(link, Outside, directory: true);

        Assert.False(ProjectPaths.IsWithinReal(Path.Combine(link, "secret.csv"), Project));
        Assert.False(ProjectPaths.IsWithinReal(Path.Combine(link, "not-yet.csv"), Project));   // a path that does not exist yet
    }

    [Fact]
    public void ALink_ThatStaysInsideTheProject_IsInside()
    {
        var link = Path.Combine(Project, "datasets", "alias.csv");
        Link(link, Path.Combine(Project, "datasets", "real.csv"), directory: false);

        Assert.True(ProjectPaths.IsWithinReal(link, Project));
    }

    [Fact]
    public void AProjectRoot_ReachedThroughALink_StillContainsItsFiles()
    {
        var alias = Path.Combine(_base, "alias");
        Link(alias, Project, directory: true);

        Assert.True(ProjectPaths.IsWithinReal(Path.Combine(alias, "datasets", "real.csv"), alias));
        Assert.True(ProjectPaths.IsWithinReal(Path.Combine(Project, "datasets", "real.csv"), alias));
    }

    [Fact]
    public void LinksThatLeadBackToThemselves_AreRefused_NotFollowedForever()
    {
        var a = Path.Combine(Project, "a");
        var b = Path.Combine(Project, "b");
        Link(a, b, directory: true);
        Link(b, a, directory: true);

        Assert.False(ProjectPaths.IsWithinReal(Path.Combine(a, "x.csv"), Project));
    }

    [Fact]
    public void WithoutLinks_RealPathIsThePath()
    {
        var file = Path.Combine(Project, "datasets", "real.csv");

        Assert.True(ProjectPaths.IsSame(file, ProjectPaths.RealPath(file)) || OperatingSystem.IsMacOS());   // /var → /private/var on macOS
        Assert.True(ProjectPaths.IsWithinReal(file, Project));
        Assert.False(ProjectPaths.IsWithinReal(Path.Combine(Outside, "secret.csv"), Project));
    }
}
