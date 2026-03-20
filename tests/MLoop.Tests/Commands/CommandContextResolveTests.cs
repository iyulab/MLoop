using MLoop.CLI.Commands;
using MLoop.CLI.Infrastructure.Configuration;

namespace MLoop.Tests.Commands;

public class CommandContextResolveTests
{
    #region ResolveModelName

    [Fact]
    public void ResolveModelName_Null_ReturnsDefault()
    {
        Assert.Equal(ConfigDefaults.DefaultModelName, CommandContext.ResolveModelName(null));
    }

    [Fact]
    public void ResolveModelName_Empty_ReturnsDefault()
    {
        Assert.Equal(ConfigDefaults.DefaultModelName, CommandContext.ResolveModelName(""));
    }

    [Fact]
    public void ResolveModelName_Whitespace_ReturnsDefault()
    {
        Assert.Equal(ConfigDefaults.DefaultModelName, CommandContext.ResolveModelName("   "));
    }

    [Fact]
    public void ResolveModelName_ValidName_TrimsAndLowercases()
    {
        Assert.Equal("my-model", CommandContext.ResolveModelName("  My-Model  "));
    }

    [Fact]
    public void ResolveModelName_AlreadyLowercase_ReturnsAsIs()
    {
        Assert.Equal("default", CommandContext.ResolveModelName("default"));
    }

    #endregion

    #region ResolveOptionalModelName

    [Fact]
    public void ResolveOptionalModelName_Null_ReturnsNull()
    {
        Assert.Null(CommandContext.ResolveOptionalModelName(null));
    }

    [Fact]
    public void ResolveOptionalModelName_Empty_ReturnsNull()
    {
        Assert.Null(CommandContext.ResolveOptionalModelName(""));
    }

    [Fact]
    public void ResolveOptionalModelName_Whitespace_ReturnsNull()
    {
        Assert.Null(CommandContext.ResolveOptionalModelName("   "));
    }

    [Fact]
    public void ResolveOptionalModelName_ValidName_TrimsAndLowercases()
    {
        Assert.Equal("test-model", CommandContext.ResolveOptionalModelName("  Test-Model  "));
    }

    #endregion
}
