using MLoop.Core.Storage;

namespace MLoop.Core.Tests.Storage;

/// <summary>An experiment id names a folder under <c>staging/</c>; one that is a path is not an experiment.</summary>
public class ExperimentFolderNameTests
{
    [Theory]
    [InlineData("exp-001")]
    [InlineData("exp-12345")]
    [InlineData("run 7")]
    public void AFolderName_IsOne(string id) => Assert.True(ExperimentLayout.IsFolderName(id));

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData(".")]
    [InlineData("..")]
    [InlineData("../..")]
    [InlineData("../exp-001")]
    [InlineData("exp-001/")]
    [InlineData("a/b")]
    public void AnythingElse_IsNot(string? id) => Assert.False(ExperimentLayout.IsFolderName(id));
}
