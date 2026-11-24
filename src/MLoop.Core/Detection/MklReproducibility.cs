using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace MLoop.Core.Detection;

/// <summary>
/// Asks MKL for bit-for-bit repeatable arithmetic, once, before this process makes its first MKL call.
/// </summary>
/// <remarks>
/// <para>
/// ML.NET's singular spectrum analysis (<c>ForecastBySsa</c>) runs on MKL, and with MKL's default (reproducibility off) the same input gave
/// different low-order bits from one fit to the next — inside one process as well as between processes. Measured on x64 on two native Windows
/// machines (one Intel, one AMD), in a native Linux container and in a translated one. Most of it is noise at the 1e-7 level, but on a series
/// whose fit sits near a discontinuity it is the difference between two different forecasts, and the width of the stored prediction band is
/// calibrated from a handful of such fits (see <c>ForecastBandCalibration</c>): on native Linux, six trainings of one random walk stored a band
/// scale of 0.5447 or 0.9070. Thread count is not the cause (one thread varied as much); MKL's own conditional numerical reproducibility
/// removes it: with <c>MKL_CBWR=AUTO</c> every fit of the four series tried was bit-identical, within and between processes, on all four.
/// </para>
/// <para>
/// <b>Why <c>AUTO</c>.</b> It keeps MKL's normal choice of instruction set and fixes what made the answer move (reductions, scheduling, cache
/// sizes), so the promise is the one MLoop makes elsewhere: the same machine and version give the same result. It does <b>not</b> make two
/// different processors agree — on one of the series an Intel machine and an AMD one stored different band scales. A user who needs the same
/// bits across processors can set <c>MKL_CBWR=COMPATIBLE</c> (the baseline code path; it gave bit-identical fits on that Intel machine, on the
/// AMD one and under instruction translation).
/// </para>
/// <para>
/// The setting is read once, when MKL initializes, so it has to be in the process environment before the first MKL call of any kind — the
/// availability probe in <see cref="TimeSeriesNativeSupport"/> is an MKL call too — and a module initializer is the one place that is
/// earlier than every caller in MLoop's own processes. A host that has already used MKL before it first touches this assembly is not
/// reached: MKL ignores the variable then, and nothing portable reads the mode back, so <see cref="Outcome.Set"/> says the variable was
/// set, not that MKL honoured it. On Windows <see cref="Environment.SetEnvironmentVariable(string, string)"/> reaches the native
/// environment; on Unix it only changes .NET's own copy, so the variable is also set with the C library's <c>setenv</c>. A value the user
/// already set is left alone — the variable is MKL's documented control, and the one who set it knows more than this does. Note that no
/// value turns reproducibility off again: MKL reads a word it does not know as <c>AUTO</c>. Where no MKL build exists (arm64) nothing is set.
/// </para>
/// <para>
/// <b>Cost and limits.</b> One forecasting training of one series, timed against the default on an AMD and on an Intel Windows machine, showed
/// no difference beyond run-to-run noise (within 4 % at the median). The setting applies to every MKL consumer in the process, not only
/// forecasting — time-series anomaly detection runs on MKL too, and neither its variation nor its speed was measured. It does not remove the
/// discontinuity itself: it makes one machine take the same side of it every time. What keeps the setting honest is the repeat-fit test
/// (<c>MklReproducibilityTests</c>): it fits ten stretches of a seeded random walk, fits each again, and compares the bits of the forecast and
/// its band.
/// </para>
/// </remarks>
public static class MklReproducibility
{
    /// <summary>MKL's documented control for conditional numerical reproducibility.</summary>
    public const string Variable = "MKL_CBWR";

    /// <summary>Reproducibility on, with MKL's normal instruction-set dispatch: the same machine gives the same bits.</summary>
    public const string Mode = "AUTO";

    /// <summary>What this process did about the setting — the one thing its value says is whether the repeatability setting stands.</summary>
    public enum Outcome
    {
        /// <summary>The module initializer has not run (only before the assembly loads — a caller never sees this).</summary>
        NotRun,

        /// <summary>This process set <see cref="Variable"/> to <see cref="Mode"/> in the native environment.</summary>
        Set,

        /// <summary>The environment already named a mode; that is the user's choice and was not touched.</summary>
        LeftToTheUser,

        /// <summary>There is no MKL for this architecture, so there is nothing to set.</summary>
        NoMklHere,

        /// <summary>Setting it failed; the process runs with MKL's default dispatch.</summary>
        Failed,
    }

    /// <summary>What <see cref="Apply"/> did in this process.</summary>
    public static Outcome Applied { get; private set; }

    // CA2255 asks a library not to run code at load. This is the exception it describes: the setting must precede every MKL call in the
    // process, no caller of this assembly is earlier, and it only sets an environment variable (never throws — see ApplyTo).
#pragma warning disable CA2255
    [ModuleInitializer]
#pragma warning restore CA2255
    internal static void Apply()
    {
        Applied = ApplyTo(Environment.GetEnvironmentVariable, SetNative, RuntimeInformation.ProcessArchitecture);
    }

    /// <summary>The decision, with the environment and the setter passed in so a test can see every branch without touching the process.</summary>
    internal static Outcome ApplyTo(Func<string, string?> read, Action<string, string> setNative, Architecture architecture)
    {
        if (architecture is not (Architecture.X64 or Architecture.X86))
            return Outcome.NoMklHere;
        if (!string.IsNullOrEmpty(read(Variable)))
            return Outcome.LeftToTheUser;
        try
        {
            setNative(Variable, Mode);
            return Outcome.Set;
        }
        catch (Exception)
        {
            // A startup type that throws takes every command down with it; the cost of this failing is only the default dispatch.
            return Outcome.Failed;
        }
    }

    internal static void SetNative(string name, string value)
    {
        Environment.SetEnvironmentVariable(name, value);
        if (!OperatingSystem.IsWindows())
            ThrowIfFailed(setenv(name, value, 1));
    }

    /// <summary>The value of <paramref name="name"/> in the native process environment, as MKL would read it — <c>null</c> when unset.</summary>
    internal static string? ReadNative(string name)
    {
        if (OperatingSystem.IsWindows())
            return Environment.GetEnvironmentVariable(name);
        var pointer = getenv(name);
        return pointer == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(pointer);
    }

    private static void ThrowIfFailed(int result)
    {
        if (result != 0)
            throw new InvalidOperationException($"setenv failed with {result}.");
    }

    // On macOS "libc" resolves to libSystem, which holds both.
    [DllImport("libc", EntryPoint = "setenv", CharSet = CharSet.Ansi)]
    private static extern int setenv(string name, string value, int overwrite);

    [DllImport("libc", EntryPoint = "getenv", CharSet = CharSet.Ansi)]
    private static extern IntPtr getenv(string name);
}
