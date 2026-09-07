using System.Globalization;

namespace MLoop.Core.Globalization;

/// <summary>
/// The single authority for the culture MLoop formats its output in.
/// </summary>
/// <remarks>
/// <para>
/// MLoop's messages are written in one language, and the numbers inside them are read by the same
/// consumers that read <c>--json</c>: a CI script, a downstream tool, a diff of two runs. Left to the
/// ambient culture, the same run prints <c>Accuracy 0.9750 … (97.5%)</c> on one machine and
/// <c>Accuracy 0,9750 … (97,5 %)</c> on another — English prose carrying another locale's decimal
/// separator, and a payload whose numbers change shape with the host. Pinning removes the host from
/// the contract: what a user reads and what a machine parses is the same everywhere.
/// </para>
/// <para>
/// Every entry point calls <see cref="Pin"/> before producing output, and the test suites pin the
/// same culture so they assert what a user actually gets. It lives here, rather than as three
/// literals, because a second place that decides this is how the three drift apart.
/// </para>
/// <para>
/// It is the invariant culture with one correction. Invariant renders a percentage as
/// <c>97.5 %</c>, and MLoop's messages are English prose where the figure reads <c>97.5%</c> — the
/// form every existing output and its tests already use. Pinning plain invariant would have traded
/// a host-dependent contract for a needlessly changed one; the correction is made here, once,
/// rather than by avoiding <c>:P</c> at each of the sites that format a percentage.
/// </para>
/// </remarks>
public static class OutputCulture
{
    private static readonly CultureInfo Pinned = CreatePinned();

    /// <summary>The culture every MLoop surface formats and parses in.</summary>
    public static CultureInfo Culture => Pinned;

    private static CultureInfo CreatePinned()
    {
        var culture = (CultureInfo)CultureInfo.InvariantCulture.Clone();
        // 1 == "n%". Invariant's default is 0 == "n %".
        culture.NumberFormat.PercentPositivePattern = 1;
        culture.NumberFormat.PercentNegativePattern = 1;
        return CultureInfo.ReadOnly(culture);
    }

    /// <summary>
    /// Pins <see cref="Culture"/> for this thread and every thread started afterwards. Call once,
    /// first thing, from an entry point.
    /// </summary>
    public static void Pin()
    {
        CultureInfo.DefaultThreadCurrentCulture = Culture;
        CultureInfo.DefaultThreadCurrentUICulture = Culture;
        CultureInfo.CurrentCulture = Culture;
        CultureInfo.CurrentUICulture = Culture;
    }
}
