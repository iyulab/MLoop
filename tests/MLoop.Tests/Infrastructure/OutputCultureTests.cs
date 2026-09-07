using System.Globalization;
using MLoop.Core.Globalization;
using Xunit;

namespace MLoop.Tests.Infrastructure;

/// <summary>
/// The output culture is a contract, not a default. These tests are the negative control for it:
/// they render under a host culture that formats differently and assert the result did not move.
/// </summary>
/// <remarks>
/// Without the pin, <c>{share:P1}</c> rendered <c>97.5%</c> on a developer machine and
/// <c>97.5 %</c> on a CI runner starting in the invariant culture — which is how a release
/// pipeline failed on an assertion no local run could reproduce. The suite is serialized
/// (<c>DisableTestParallelization</c>), so the culture these tests move is theirs alone.
/// </remarks>
public class OutputCultureTests
{
    [Theory]
    [InlineData("de-DE")]
    [InlineData("fr-FR")]
    [InlineData("ko-KR")]
    [InlineData("")] // the invariant culture a runner starts in
    public void Pin_overrides_whatever_culture_the_host_started_in(string hostCulture)
    {
        var restore = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = hostCulture.Length == 0
                ? CultureInfo.InvariantCulture
                : CultureInfo.GetCultureInfo(hostCulture);

            OutputCulture.Pin();

            // Ambient interpolation — what every message-building site in the product does.
            Assert.Equal("0.9750", $"{0.975:F4}");
            Assert.Equal("97.5%", $"{0.975:P1}");
            Assert.Equal("1234.5", $"{1234.5:0.#}");
        }
        finally
        {
            CultureInfo.CurrentCulture = restore;
        }
    }

    [Fact]
    public void Percent_is_rendered_the_way_english_prose_writes_it()
    {
        // The one place MLoop departs from the invariant culture, stated where it can be checked:
        // invariant renders "97.5 %", and every message and its tests use "97.5%".
        Assert.Equal("97.5 %", 0.975.ToString("P1", CultureInfo.InvariantCulture));
        Assert.Equal("97.5%", 0.975.ToString("P1", OutputCulture.Culture));
    }

    [Fact]
    public void Decimal_separator_does_not_follow_the_host()
    {
        // The defect this guards: an English message carrying another locale's separator.
        Assert.Equal("0,9750", 0.975.ToString("F4", CultureInfo.GetCultureInfo("de-DE")));
        Assert.Equal("0.9750", 0.975.ToString("F4", OutputCulture.Culture));
    }
}
