using FluentAssertions;

namespace MLoop.API.Tests;

/// <summary>A job's answer speaks in the project's paths, not in the layout of the host that runs it.</summary>
public class ProjectRelativeTextTests
{
    private const string Root = "/srv/work/proj";

    [Theory]
    [InlineData("/srv/work/proj/datasets/images", "datasets/images")]
    [InlineData("No class subfolders found in '/srv/work/proj/datasets/x'. Image classification requires…", "No class subfolders found in 'datasets/x'. Image classification requires…")]
    [InlineData("kept at /srv/work/proj", "kept at .")]
    [InlineData("'/srv/work/proj'", "'.'")]
    public void ThePrefixOfTheProjectRoot_IsTakenOffThePathsUnderIt(string text, string expected)
        => ProjectRelativeText.In(text, Root).Should().Be(expected);

    [Fact]
    public void ASiblingThatOnlyStartsWithTheRootsName_IsNotTheProject()
        => ProjectRelativeText.In("/srv/work/proj-other/datasets/x", Root).Should().Be("/srv/work/proj-other/datasets/x");

    [Fact]
    public void APathOutsideTheProject_IsLeftAlone()
        => ProjectRelativeText.In("Could not read /etc/ssl/openssl.cnf", Root).Should().Be("Could not read /etc/ssl/openssl.cnf");

    [Fact]
    public void ARootWrittenWithBackslashes_IsTakenOffInEitherSeparator()
    {
        ProjectRelativeText.In(@"D:\work\proj\datasets\x", @"D:\work\proj").Should().Be(@"datasets\x");
        ProjectRelativeText.In("D:/work/proj/datasets/x", @"D:\work\proj").Should().Be("datasets/x");
    }

    [Fact]
    public void NothingToSay_StaysNothing()
    {
        ProjectRelativeText.In((string?)null, Root).Should().BeNull();
        ProjectRelativeText.In("", Root).Should().BeEmpty();
        ProjectRelativeText.In((IReadOnlyList<string>?)null, Root).Should().BeNull();
    }
}
