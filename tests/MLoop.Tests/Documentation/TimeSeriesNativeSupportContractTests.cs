using System.Text.RegularExpressions;
using MLoop.Core.Detection;

namespace MLoop.Tests.Documentation;

/// <summary>
/// Whether this machine can run ML.NET's time-series algorithms is decided in exactly one place, and
/// that place is product code rather than a test.
/// </summary>
/// <remarks>
/// <para>
/// This condition was known to the repository long before it had an authority: four test projects
/// carried five independent copies of a <c>MklAvailable</c> probe, each re-deriving both the probe
/// and the predicate for "the native failed to load". Nothing in the product knew it, so a user on
/// Linux was shown <c>The type initializer for 'FftUtils' threw an exception</c> — a sentence naming
/// neither the missing runtime nor a way to supply it.
/// </para>
/// <para>
/// The copies then failed the way copies fail. A sixth site, added later, simply did not have one:
/// its <c>detect</c> case asserted a successful exit unconditionally, and every Linux and macOS CI
/// run failed on it for a month while the local release gate — which is Windows, where the native is
/// self-contained — stayed green. That is what these two scans exist to prevent: not the condition,
/// which is upstream's and legitimate, but a seventh copy of the knowledge about it.
/// </para>
/// </remarks>
public class TimeSeriesNativeSupportContractTests
{
    /// <summary>
    /// A test that decides for itself whether the native loaded, instead of asking the authority.
    /// The old copies all took this shape: catch the loader's exception and turn it into a bool.
    /// </summary>
    private static readonly Regex LocalNativeProbe =
        new(@"catch\s*\([^)]*Exception[^)]*\)\s*when\s*\([^)]*DllNotFoundException", RegexOptions.Compiled);

    /// <summary>A locally-held answer to "is the native here", whatever it is named.</summary>
    private static readonly Regex LocalAvailabilityFlag =
        new(@"Lazy<bool>\s+\w*(?:Mkl|Native|Fft)\w*\s*=", RegexOptions.Compiled);

    private static bool ReDerivesNativeAvailability(string source) =>
        LocalNativeProbe.IsMatch(source) || LocalAvailabilityFlag.IsMatch(source);

    [Fact]
    public void NoTestDecidesForItselfWhetherTheTimeSeriesNativeLoaded()
    {
        var scanned = RepoSourceTree.SourceFiles(nameof(TimeSeriesNativeSupportContractTests)).ToList();

        // The scan reaches the suites this guard is about. Without this, a moved directory or a
        // renamed project would empty the set, and an empty set reports exactly what compliance
        // reports.
        Assert.Contains(scanned, f => Path.GetFileName(f) == "SrCnnOneShotDetectorTests.cs");
        Assert.Contains(scanned, f => Path.GetFileName(f) == "PredictForecastingTests.cs");
        Assert.Contains(scanned, f => Path.GetFileName(f) == "ForecastingApiTests.cs");
        Assert.Contains(scanned, f => Path.GetFileName(f) == "PredictionServiceTests.cs");

        var offenders = scanned
            .Where(f => ReDerivesNativeAvailability(File.ReadAllText(f)))
            .Select(RepoSourceTree.Relative)
            .Order(StringComparer.Ordinal)
            .ToList();

        Assert.True(offenders.Count == 0,
            "Whether ML.NET's time-series native loaded is answered by "
            + nameof(TimeSeriesNativeSupport) + "." + nameof(TimeSeriesNativeSupport.IsAvailable)
            + ", so the platform condition is described once and the product describes it too:\n"
            + string.Join('\n', offenders));
    }

    [Fact]
    public void TheScanSeesTheShapeItForbidsAndAcceptsTheOneItWants()
    {
        // Companion to the scan above. Five copies existed and now none do, so from here on the scan
        // reports "clean" whether or not its patterns still match anything — and those two outcomes
        // are indistinguishable without feeding it the shape it exists to catch.
        Assert.True(ReDerivesNativeAvailability("""
            private static readonly Lazy<bool> MklAvailable = new(() =>
            {
                try { Probe(); return true; }
                catch (Exception ex) when (ex is DllNotFoundException or TypeInitializationException
                                           || ex.InnerException is DllNotFoundException)
                { return false; }
            });
            """));

        // A renamed copy is still a copy: the flag pattern catches it even where the catch filter
        // has been reworded.
        Assert.True(ReDerivesNativeAvailability(
            "private static readonly Lazy<bool> FftNativePresent = new(Probe);"));

        // Consuming the authority is the shape this guard wants, and must never be reported.
        Assert.False(ReDerivesNativeAvailability(
            "if (!TimeSeriesNativeSupport.IsAvailable) return;"));

        // Nor may an ordinary catch of an unrelated exception look like a probe.
        Assert.False(ReDerivesNativeAvailability(
            "catch (Exception ex) when (ex is InvalidOperationException) { return false; }"));
    }

    [Fact]
    public void ThePredicateAnswersForTheExceptionTheLoaderActuallyThrows()
    {
        // Measured shape: TypeInitializationException('FftUtils') wrapping DllNotFoundException.
        var loaderFailure = new TypeInitializationException(
            "Microsoft.ML.Transforms.TimeSeries.FftUtils",
            new DllNotFoundException("Unable to load shared library 'MklImports' ... libiomp5.so"));

        Assert.True(TimeSeriesNativeSupport.IsMissingNative(loaderFailure));
        Assert.True(TimeSeriesNativeSupport.IsMissingNative(
            new InvalidOperationException("wrapped", loaderFailure)));

        // A static constructor that failed for its own reasons is not a missing native, and saying
        // it is would send a user off installing a package that was never the problem. The predicate
        // this replaced answered true here.
        Assert.False(TimeSeriesNativeSupport.IsMissingNative(
            new TypeInitializationException("Some.Other.Type", new InvalidOperationException("bad config"))));
        Assert.False(TimeSeriesNativeSupport.IsMissingNative(new InvalidOperationException("unrelated")));
    }

    [Fact]
    public void TheMessageNamesWhatIsMissingAndWhatToDoAboutIt()
    {
        // The whole point of the authority owning a sentence: a loader error names a type
        // initializer, which no user can act on.
        Assert.Contains("libiomp5", TimeSeriesNativeSupport.UnavailableMessage, StringComparison.Ordinal);
        Assert.Contains("libomp5", TimeSeriesNativeSupport.UnavailableMessage, StringComparison.Ordinal);
        Assert.Contains("arm64", TimeSeriesNativeSupport.UnavailableMessage, StringComparison.Ordinal);
    }
}
