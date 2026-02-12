using MLoop.CLI.Infrastructure.Update;

namespace MLoop.Tests.Infrastructure.Update;

public class UpdateCheckerTests
{
    // --- CompareVersions tests ---

    [Theory]
    [InlineData("1.0.0", "0.9.0", 1)]       // major higher
    [InlineData("0.2.0", "0.1.0", 1)]        // minor higher
    [InlineData("0.1.1", "0.1.0", 1)]        // patch higher
    [InlineData("0.1.0", "0.1.0", 0)]        // equal
    [InlineData("0.1.0", "0.2.0", -1)]       // lower
    [InlineData("1.0.0", "1.0.0", 0)]        // equal
    public void CompareVersions_NumericVersions(string a, string b, int expectedSign)
    {
        var result = UpdateChecker.CompareVersions(a, b);
        Assert.Equal(expectedSign, Math.Sign(result));
    }

    [Theory]
    [InlineData("1.0.0", "1.0.0-alpha", 1)]   // release > prerelease
    [InlineData("1.0.0-alpha", "1.0.0", -1)]   // prerelease < release
    [InlineData("1.0.0-beta", "1.0.0-alpha", 1)]  // beta > alpha (string compare)
    [InlineData("1.0.0-alpha", "1.0.0-alpha", 0)]  // same prerelease
    public void CompareVersions_PrereleaseVersions(string a, string b, int expectedSign)
    {
        var result = UpdateChecker.CompareVersions(a, b);
        Assert.Equal(expectedSign, Math.Sign(result));
    }

    [Fact]
    public void CompareVersions_DifferentPartCount()
    {
        // 1.0 is treated as 1.0.0
        var result = UpdateChecker.CompareVersions("1.0", "1.0.0");
        Assert.Equal(0, result);
    }

    [Fact]
    public void CompareVersions_LongerVersion()
    {
        var result = UpdateChecker.CompareVersions("1.0.0.1", "1.0.0.0");
        Assert.True(result > 0);
    }

    // --- GetCurrentVersion tests ---

    [Fact]
    public void GetCurrentVersion_ReturnsNonEmptyString()
    {
        var version = UpdateChecker.GetCurrentVersion();
        Assert.False(string.IsNullOrWhiteSpace(version));
    }

    [Fact]
    public void GetCurrentVersion_DoesNotContainPlusCommitHash()
    {
        var version = UpdateChecker.GetCurrentVersion();
        Assert.DoesNotContain("+", version);
    }

    // --- UpdateInfo tests ---

    [Fact]
    public void UpdateInfo_RecordEquality()
    {
        var info1 = new UpdateInfo("1.0.0", "0.9.0", true);
        var info2 = new UpdateInfo("1.0.0", "0.9.0", true);
        Assert.Equal(info1, info2);
    }

    [Fact]
    public void UpdateInfo_Properties()
    {
        var info = new UpdateInfo("2.0.0", "1.0.0", true);
        Assert.Equal("2.0.0", info.LatestVersion);
        Assert.Equal("1.0.0", info.CurrentVersion);
        Assert.True(info.UpdateAvailable);
    }
}
