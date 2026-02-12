using MLoop.CLI.Infrastructure.Update;

namespace MLoop.Tests.Infrastructure.Update;

public class InstallDetectorTests
{
    [Fact]
    public void GetRuntimeIdentifier_ReturnsValidFormat()
    {
        var rid = InstallDetector.GetRuntimeIdentifier();

        Assert.NotNull(rid);
        Assert.Contains("-", rid); // e.g., "win-x64", "linux-arm64"
    }

    [Fact]
    public void GetRuntimeIdentifier_StartsWithKnownOs()
    {
        var rid = InstallDetector.GetRuntimeIdentifier();

        var os = rid.Split('-')[0];
        Assert.Contains(os, new[] { "win", "osx", "linux" });
    }

    [Fact]
    public void GetRuntimeIdentifier_EndsWithKnownArch()
    {
        var rid = InstallDetector.GetRuntimeIdentifier();

        var arch = rid.Split('-')[1];
        Assert.Contains(arch, new[] { "x64", "arm64" });
    }

    [Fact]
    public void Detect_ReturnsValidEnumValue()
    {
        var method = InstallDetector.Detect();

        Assert.True(Enum.IsDefined(typeof(InstallMethod), method));
    }
}
