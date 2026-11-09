using System.Runtime.InteropServices;
using MLoop.Core.Detection;

namespace MLoop.Tests.Documentation;

/// <summary>
/// The time-series native is missing for two different reasons and the way out differs: on Linux x64 the system supplies an OpenMP
/// runtime (install steps), on an arm64 machine there is no build to load (nothing to install — use another machine). One sentence
/// carrying both told a Mac user to run <c>apt-get</c> first and never said where the training can run.
/// </summary>
public sealed class TimeSeriesNativeMessageForThisMachineTests
{
    [Fact]
    public void An_arm64_machine_is_told_the_platform_cannot_and_where_to_run_instead()
    {
        var message = TimeSeriesNativeSupport.UnavailableFor("macOS", Architecture.Arm64);

        Assert.StartsWith(TimeSeriesNativeSupport.UnavailableLead, message, StringComparison.Ordinal);
        Assert.Contains("macOS on arm64", message, StringComparison.Ordinal);
        Assert.Contains("Linux x64 machine or container", message, StringComparison.Ordinal);
        Assert.Contains("mloop detect <data-file>", message, StringComparison.Ordinal);
        Assert.DoesNotContain("apt-get", message, StringComparison.Ordinal);
    }

    [Fact]
    public void Linux_x64_is_told_the_install_steps_and_nothing_about_other_platforms()
    {
        var message = TimeSeriesNativeSupport.UnavailableFor("Linux", Architecture.X64);

        Assert.StartsWith(TimeSeriesNativeSupport.UnavailableLead, message, StringComparison.Ordinal);
        Assert.Contains("apt-get update && apt-get install -y libomp5", message, StringComparison.Ordinal);
        Assert.Contains("libiomp5.so", message, StringComparison.Ordinal);
        Assert.DoesNotContain("arm64", message, StringComparison.Ordinal);
    }

    [Fact]
    public void Any_other_machine_gets_both_causes_and_every_variant_starts_with_the_same_lead()
    {
        var other = TimeSeriesNativeSupport.UnavailableFor("Windows", Architecture.X64);

        Assert.Equal(TimeSeriesNativeSupport.UnavailableMessage, other);
        Assert.StartsWith(TimeSeriesNativeSupport.UnavailableLead, TimeSeriesNativeSupport.UnavailableMessage, StringComparison.Ordinal);
        Assert.False(string.IsNullOrWhiteSpace(TimeSeriesNativeSupport.UnavailableHere));
    }
}
