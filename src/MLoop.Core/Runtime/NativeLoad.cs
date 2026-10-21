using System.Runtime.InteropServices;

namespace MLoop.Core.Runtime;

/// <summary>
/// A native library that does not load on this machine — told apart from an error of the algorithm
/// that would have used it.
/// </summary>
public static class NativeLoad
{
    /// <summary>
    /// True when a <see cref="DllNotFoundException"/> is <paramref name="ex"/> or anywhere in its
    /// inner chain (ML.NET wraps it in a type-initializer failure). Any other type-initializer
    /// failure is a real bug, not a missing native, and is not counted.
    /// </summary>
    public static bool IsMissing(Exception ex)
    {
        for (var e = ex; e is not null; e = e.InnerException)
        {
            if (e is DllNotFoundException)
                return true;
        }

        return false;
    }

    /// <summary>
    /// One line for a trainer left out because its native library does not load here — the loader's
    /// own message lists every path it tried and runs to dozens of lines.
    /// </summary>
    public static string TrainerSkipped(string trainerName) =>
        $"{trainerName} skipped: its native library does not load on {RuntimeInformation.RuntimeIdentifier}.";
}
