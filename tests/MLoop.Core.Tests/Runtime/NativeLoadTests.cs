using System.Runtime.InteropServices;
using MLoop.Core.Runtime;

namespace MLoop.Core.Tests.Runtime;

public class NativeLoadTests
{
    [Fact]
    public void A_missing_library_anywhere_in_the_chain_counts()
    {
        Assert.True(NativeLoad.IsMissing(new DllNotFoundException("lib_lightgbm")));
        Assert.True(NativeLoad.IsMissing(new TypeInitializationException("X", new DllNotFoundException("lib"))));
        Assert.True(NativeLoad.IsMissing(new InvalidOperationException("outer", new TypeInitializationException("X", new DllNotFoundException("lib")))));
    }

    [Fact]
    public void Another_failure_is_not_a_missing_library()
    {
        Assert.False(NativeLoad.IsMissing(new TypeInitializationException("X", new NullReferenceException())));
        Assert.False(NativeLoad.IsMissing(new InvalidOperationException("invalid label")));
    }

    [Fact]
    public void A_skipped_trainer_is_one_line_naming_the_platform()
    {
        var line = NativeLoad.TrainerSkipped("LightGbmRanking");

        Assert.DoesNotContain('\n', line);
        Assert.StartsWith("LightGbmRanking skipped", line);
        Assert.Contains(RuntimeInformation.RuntimeIdentifier, line);
    }
}
