using MLoop.CLI.Infrastructure.Diagnostics;

namespace MLoop.Tests.Diagnostics;

/// <summary>A failed run's text carries the cause a library wrapped — the wrapper's message alone names nothing a caller can act on.</summary>
public class FailureCauseTests
{
    [Fact]
    public void AWrappedCause_FollowsTheWrappersMessage()
    {
        var ex = new InvalidOperationException("Splitter/consolidator worker encountered exception while consuming source data",
            new InvalidDataException("Image 'img005.png' could not be decoded"));

        Assert.Equal("Splitter/consolidator worker encountered exception while consuming source data — cause: Image 'img005.png' could not be decoded",
            FailureCause.Describe(ex));
    }

    [Fact]
    public void TheInnermostOfSeveralLevels_IsTheCause()
    {
        var ex = new InvalidOperationException("outer", new AggregateException("agg",
            new InvalidOperationException("middle", new FormatException("the actual reason"))));

        Assert.EndsWith("— cause: the actual reason", FailureCause.Describe(ex), StringComparison.Ordinal);
    }

    [Fact]
    public void ACauseTheMessageAlreadyCarries_IsNotRepeated()
    {
        var ex = new InvalidOperationException("Training failed: label is empty", new ArgumentException("label is empty"));

        Assert.Equal("Training failed: label is empty", FailureCause.Describe(ex));
    }

    [Fact]
    public void NoInnerException_IsJustTheMessage()
        => Assert.Equal("plain", FailureCause.Describe(new InvalidOperationException("plain")));

    [Fact]
    public void ALibraryThatDoesNotLoad_KeepsTheMessageWrittenForIt_NotTheLoadersPathList()
    {
        var loader = new DllNotFoundException("libiomp5.so: cannot open shared object file: No such file or directory\n"
            + "/usr/share/dotnet/shared/libiomp5.so: cannot open shared object file\n/lib/x86_64-linux-gnu/libiomp5.so: cannot open shared object file");
        var ex = new PlatformNotSupportedException("Time-series algorithms need ML.NET's MKL native library; install the OpenMP runtime.", loader);

        Assert.Equal("Time-series algorithms need ML.NET's MKL native library; install the OpenMP runtime.", FailureCause.Describe(ex));
    }
}
