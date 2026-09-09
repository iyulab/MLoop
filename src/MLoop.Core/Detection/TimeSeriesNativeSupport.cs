using Microsoft.ML;
using Microsoft.ML.TimeSeries;

namespace MLoop.Core.Detection;

/// <summary>
/// Single authority for "can this machine run ML.NET's time-series algorithms" — the probe, the
/// exception predicate, and the sentence a user is shown when the answer is no.
/// </summary>
/// <remarks>
/// <para>
/// Every time-series surface in ML.NET — <c>ForecastBySsa</c>, <c>DetectAnomalyBySrCnn</c>,
/// <c>DetectSpikeBySsa</c>, <c>DetectEntireAnomalyBySrCnn</c>, <c>DetectSeasonality</c> — routes its
/// FFT through <c>Microsoft.ML.Transforms.TimeSeries.FftUtils</c>, which P/Invokes the
/// <c>MklImports</c> native library. That library is shipped by <c>Microsoft.ML.Mkl.Redist</c>, and
/// what it ships is not the same on every platform:
/// </para>
/// <list type="bullet">
/// <item><description>
/// <b>Windows</b> — <c>MklImports.dll</c> plus the Intel OpenMP runtime it links against
/// (<c>libiomp5md.dll</c>). Self-contained; everything works out of the box.
/// </description></item>
/// <item><description>
/// <b>Linux</b> — <c>libMklImports.so</c> ships, but the OpenMP runtime it links against does
/// <i>not</i>, and that is upstream's stated intent rather than an oversight: ML.NET's own build
/// notes say not to copy <c>libiomp5</c> on Linux because it "relies on OpenMP to be installed on
/// the system". Loading fails with <c>libiomp5.so: cannot open shared object file</c> until one is.
/// LLVM's runtime is ABI-compatible, which is why installing <c>libomp5</c> and exposing it as
/// <c>libiomp5.so</c> resolves it (measured). So this is a system prerequisite this project never
/// documented, not a bug to report upstream.
/// </description></item>
/// <item><description>
/// <b>macOS on Apple silicon</b> — no <c>osx-arm64</c> native is published at all, so there is
/// nothing to load and nothing to install. Time-series algorithms are simply unavailable there.
/// </description></item>
/// </list>
/// <para>
/// Before this type the knowledge lived as five independent copies of a <c>MklAvailable</c> probe in
/// four test projects, and product code had none — so a user on Linux was shown
/// <c>The type initializer for 'FftUtils' threw an exception</c>, which names neither the missing
/// runtime nor the way to supply it. A sixth site then guarded nothing at all, because a copy is
/// only added by someone who already knows to add it: that omission is what turned a known platform
/// condition into a red CI run that blocked a release.
/// </para>
/// </remarks>
public static class TimeSeriesNativeSupport
{
    /// <summary>
    /// What a user is told when the native is unavailable. Names the missing runtime and the action
    /// that supplies it, because the underlying loader error names only a type initializer.
    /// </summary>
    public const string UnavailableMessage =
        "Time-series algorithms (forecasting, time-series anomaly detection, detect) need ML.NET's "
        + "MKL native library, and it could not be loaded here. On Linux that library expects an "
        + "Intel-ABI OpenMP runtime to be installed on the system — ML.NET deliberately does not "
        + "bundle one — so install it and expose it under the name it looks for: "
        + "'apt-get install -y libomp5', then symlink libiomp5.so to libomp.so.5. On macOS running "
        + "Apple silicon, ML.NET publishes no osx-arm64 build of this library at all, so these "
        + "algorithms are unavailable there and no install resolves it.";

    /// <summary>
    /// True when <paramref name="ex"/> is the native library failing to load, rather than a genuine
    /// error from the algorithm.
    /// </summary>
    /// <remarks>
    /// A <see cref="TypeInitializationException"/> counts only when a <see cref="DllNotFoundException"/>
    /// is what it wraps. The predicate this replaced accepted <i>any</i> type-initializer failure,
    /// which would also have swallowed an unrelated static-constructor bug and reported it to the
    /// user as a missing native.
    /// </remarks>
    public static bool IsMissingNative(Exception ex)
    {
        for (var e = ex; e is not null; e = e.InnerException)
        {
            if (e is DllNotFoundException)
                return true;
        }

        return false;
    }

    /// <summary>
    /// Whether the native loads on this machine. Probed once, through the same
    /// <c>FftUtils</c> entry point every time-series algorithm uses, so a positive answer means all
    /// of them can run and a negative one means none can.
    /// </summary>
    public static bool IsAvailable => s_available.Value;

    private static readonly Lazy<bool> s_available = new(Probe);

    private static bool Probe()
    {
        try
        {
            // The shortest path into FftUtils: DetectSeasonality computes a forward FFT and nothing
            // else native. A series of MinimumPoints is enough to reach it.
            var ml = new MLContext(seed: 0);
            var data = ml.Data.LoadFromEnumerable(
                Enumerable.Range(0, SrCnnOneShotDetector.MinimumPoints)
                          .Select(i => new ProbePoint { Value = i % 3 }));
            ml.AnomalyDetection.DetectSeasonality(data, nameof(ProbePoint.Value));
            return true;
        }
        catch (Exception ex) when (IsMissingNative(ex))
        {
            return false;
        }
    }

    private sealed class ProbePoint
    {
        public double Value { get; set; }
    }

    /// <summary>
    /// Runs <paramref name="action"/>, translating a missing-native failure into a
    /// <see cref="PlatformNotSupportedException"/> carrying <see cref="UnavailableMessage"/>. Every
    /// other exception passes through untouched.
    /// </summary>
    public static T Guard<T>(Func<T> action)
    {
        try
        {
            return action();
        }
        catch (Exception ex) when (IsMissingNative(ex))
        {
            throw new PlatformNotSupportedException(UnavailableMessage, ex);
        }
    }

    /// <inheritdoc cref="Guard{T}(Func{T})"/>
    public static void Guard(Action action)
    {
        Guard<object?>(() =>
        {
            action();
            return null;
        });
    }
}
