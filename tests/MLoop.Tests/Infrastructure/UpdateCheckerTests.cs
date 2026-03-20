using MLoop.CLI.Infrastructure.Update;

namespace MLoop.Tests.Infrastructure;

public class UpdateCheckerTests
{
    #region CompareVersions

    [Theory]
    [InlineData("1.0.0", "0.9.0", 1)]
    [InlineData("0.10.0", "0.9.0", 1)]
    [InlineData("0.10.1", "0.10.0", 1)]
    [InlineData("0.9.0", "1.0.0", -1)]
    [InlineData("1.0.0", "1.0.0", 0)]
    [InlineData("0.10.1", "0.10.1", 0)]
    public void CompareVersions_NumericComparison(string a, string b, int expectedSign)
    {
        var result = UpdateChecker.CompareVersions(a, b);
        Assert.Equal(expectedSign, Math.Sign(result));
    }

    [Theory]
    [InlineData("1.0.0", "1.0.0-alpha", 1)]      // release > prerelease
    [InlineData("1.0.0-alpha", "1.0.0", -1)]      // prerelease < release
    [InlineData("1.0.0-beta", "1.0.0-alpha", 1)]  // beta > alpha (string comparison)
    [InlineData("1.0.0-alpha", "1.0.0-alpha", 0)]
    public void CompareVersions_PrereleaseHandling(string a, string b, int expectedSign)
    {
        var result = UpdateChecker.CompareVersions(a, b);
        Assert.Equal(expectedSign, Math.Sign(result));
    }

    [Fact]
    public void CompareVersions_DifferentLengthParts()
    {
        // 1.0 vs 1.0.0 should be equal
        Assert.Equal(0, UpdateChecker.CompareVersions("1.0", "1.0.0"));

        // 1.0.1 > 1.0
        Assert.True(UpdateChecker.CompareVersions("1.0.1", "1.0") > 0);
    }

    #endregion

    #region GetCurrentVersion

    [Fact]
    public void GetCurrentVersion_ReturnsNonEmpty()
    {
        var version = UpdateChecker.GetCurrentVersion();
        Assert.False(string.IsNullOrWhiteSpace(version));
    }

    [Fact]
    public void GetCurrentVersion_DoesNotContainPlusHash()
    {
        var version = UpdateChecker.GetCurrentVersion();
        Assert.DoesNotContain("+", version);
    }

    #endregion
}
