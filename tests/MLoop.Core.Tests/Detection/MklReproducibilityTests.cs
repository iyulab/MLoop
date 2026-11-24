using System.Runtime.InteropServices;
using Microsoft.ML;
using MLoop.Core.Detection;
using MLoop.Core.Training;
using Xunit;

namespace MLoop.Core.Tests.Detection;

/// <summary>
/// The repeatability setting for MKL (<see cref="MklReproducibility"/>): every branch of the decision, the one thing .NET does not do on Unix
/// (reach the native environment), and — where the native loads — the property it exists for: one input fitted again gives the same bits.
/// </summary>
public class MklReproducibilityTests
{
    private static readonly Func<string, string?> Unset = _ => null;
    private static readonly Action<string, string> Ignore = (_, _) => { };

    [Fact]
    public void Where_the_variable_is_unset_it_asks_for_repeatable_arithmetic_on_the_normal_dispatch()
    {
        var set = new List<(string Name, string Value)>();
        var outcome = MklReproducibility.ApplyTo(Unset, (n, v) => set.Add((n, v)), Architecture.X64);

        Assert.Equal(MklReproducibility.Outcome.Set, outcome);
        Assert.Equal([("MKL_CBWR", "AUTO")], set);
    }

    [Theory]
    [InlineData("AUTO")]
    [InlineData("AVX2")]
    [InlineData("COMPATIBLE")]
    public void A_mode_the_user_already_named_is_left_alone(string theirs)
    {
        var set = new List<string>();
        var outcome = MklReproducibility.ApplyTo(n => n == "MKL_CBWR" ? theirs : null, (n, _) => set.Add(n), Architecture.X64);

        Assert.Equal(MklReproducibility.Outcome.LeftToTheUser, outcome);
        Assert.Empty(set);
    }

    [Theory]
    [InlineData(Architecture.Arm64)]
    [InlineData(Architecture.Arm)]
    public void Where_there_is_no_mkl_build_nothing_is_set(Architecture architecture)
    {
        var set = new List<string>();
        var outcome = MklReproducibility.ApplyTo(Unset, (n, _) => set.Add(n), architecture);

        Assert.Equal(MklReproducibility.Outcome.NoMklHere, outcome);
        Assert.Empty(set);
    }

    [Fact]
    public void A_setter_that_throws_leaves_the_process_running_on_the_default_dispatch()
    {
        var outcome = MklReproducibility.ApplyTo(Unset, (_, _) => throw new InvalidOperationException("no"), Architecture.X64);

        Assert.Equal(MklReproducibility.Outcome.Failed, outcome);
    }

    [Fact]
    public void The_native_setter_reaches_the_environment_native_code_reads()
    {
        const string name = "MLOOP_TEST_NATIVE_ENV";
        try
        {
            MklReproducibility.SetNative(name, "from-the-test");
            // MKL reads the process environment through the C library, not through .NET's copy of it.
            Assert.Equal("from-the-test", MklReproducibility.ReadNative(name));
        }
        finally
        {
            Environment.SetEnvironmentVariable(name, null);
        }
    }

    [Fact]
    public void The_reader_can_tell_a_variable_that_only_dotnet_knows_from_a_native_one()
    {
        // The reason the setter calls the C library at all: on Unix, Environment.SetEnvironmentVariable never reaches it. If this stops
        // being true the C-library call is dead weight; if the reader cannot see the difference the test above proves nothing.
        Assert.SkipWhen(OperatingSystem.IsWindows(), "On Windows .NET's setter is the native one.");
        const string name = "MLOOP_TEST_DOTNET_ONLY_ENV";
        try
        {
            Environment.SetEnvironmentVariable(name, "managed-only");
            Assert.Null(MklReproducibility.ReadNative(name));
        }
        finally
        {
            Environment.SetEnvironmentVariable(name, null);
        }
    }

    [Fact]
    public void The_module_initializer_has_run_in_this_process()
    {
        Assert.NotEqual(MklReproducibility.Outcome.NotRun, MklReproducibility.Applied);
        if (MklReproducibility.Applied == MklReproducibility.Outcome.Set)
            Assert.Equal("AUTO", MklReproducibility.ReadNative("MKL_CBWR"));
    }

    /// <summary>
    /// The property the setting is for. Without it a second fit of one input differs from the first in the low bits (measured on every series tried,
    /// the walk and the noise alike) — but not when the same arrays are fitted back to back: the difference shows when fits of different lengths are
    /// interleaved, which is how the band calibration runs them. So this fits the stretches of the calibration's own schedule (one horizon shorter each)
    /// and then each of them again, and every second fit must equal its first, bit for bit.
    /// </summary>
    [TimeSeriesNativeFact]
    public void A_stretch_fitted_again_after_others_gives_the_same_bits()
    {
        Assert.SkipWhen(
            MklReproducibility.Applied != MklReproducibility.Outcome.Set && Environment.GetEnvironmentVariable("MKL_CBWR") is not ("AUTO" or "COMPATIBLE"),
            "MKL_CBWR names another mode in this environment: the user's choice, whose repeatability this test does not vouch for.");

        const int length = 2000, horizon = 10, stretches = 10;
        var window = ForecastingWindow.Default(length);
        var random = new Random(20261124);
        var series = new float[length];
        double level = 0;
        for (var i = 0; i < length; i++)
        {
            level += random.NextDouble() - 0.5;
            series[i] = (float)level;
        }

        var ml = new MLContext(seed: 1);
        var first = new ForecastBandBacktest.Forecast[stretches];
        for (var k = 0; k < stretches; k++)
            first[k] = ForecastBandBacktest.FitAndForecast(ml, series, 0, length - k * horizon, horizon, window, 0);
        for (var k = 0; k < stretches; k++)
        {
            var again = ForecastBandBacktest.FitAndForecast(ml, series, 0, length - k * horizon, horizon, window, 0);
            Assert.True(Bits(first[k].Values).SequenceEqual(Bits(again.Values)), $"stretch {k}: the forecast moved");
            Assert.True(Bits(first[k].Lower).SequenceEqual(Bits(again.Lower)), $"stretch {k}: the lower bound moved");
            Assert.True(Bits(first[k].Upper).SequenceEqual(Bits(again.Upper)), $"stretch {k}: the upper bound moved");
        }
    }

    private static int[] Bits(float[] values) => values.Select(BitConverter.SingleToInt32Bits).ToArray();
}
