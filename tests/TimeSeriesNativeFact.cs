using System.Runtime.CompilerServices;
using MLoop.Core.Detection;
using Xunit;

/// <summary>
/// A fact that needs ML.NET's time-series native (MKL FFT). Where the native is absent — osx-arm64
/// always, Linux without an Intel-ABI OpenMP runtime — the test is reported as skipped, with the
/// product's own explanation as the reason.
/// </summary>
/// <remarks>
/// These tests used to return early when the native was unavailable. That reported a pass for a
/// test that ran nothing, so a machine with no time-series coverage at all
/// showed the same green as one with full coverage. A skip says what was not checked.
/// </remarks>
public sealed class TimeSeriesNativeFactAttribute : FactAttribute
{
    // The caller's file and line reach the base so a skipped or failed test still points at its source.
    public TimeSeriesNativeFactAttribute(
        [CallerFilePath] string? sourceFilePath = null,
        [CallerLineNumber] int sourceLineNumber = -1)
        : base(sourceFilePath, sourceLineNumber)
    {
        if (!TimeSeriesNativeSupport.IsAvailable)
            Skip = TimeSeriesNativeSupport.UnavailableMessage;
    }
}
