using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

public class CommandContextTests
{
    // --- ResolveModelName tests ---

    [Fact]
    public void ResolveModelName_Null_ReturnsDefault()
    {
        Assert.Equal("default", CommandContext.ResolveModelName(null));
    }

    [Fact]
    public void ResolveModelName_Empty_ReturnsDefault()
    {
        Assert.Equal("default", CommandContext.ResolveModelName(""));
    }

    [Fact]
    public void ResolveModelName_Whitespace_ReturnsDefault()
    {
        Assert.Equal("default", CommandContext.ResolveModelName("   "));
    }

    [Fact]
    public void ResolveModelName_MixedCase_ReturnsLowercase()
    {
        Assert.Equal("mymodel", CommandContext.ResolveModelName("MyModel"));
    }

    [Fact]
    public void ResolveModelName_WithSpaces_TrimsAndLowercases()
    {
        Assert.Equal("my-model", CommandContext.ResolveModelName("  My-Model  "));
    }

    // --- ResolveOptionalModelName tests ---

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
    public void ResolveOptionalModelName_MixedCase_ReturnsLowercase()
    {
        Assert.Equal("mymodel", CommandContext.ResolveOptionalModelName("MyModel"));
    }

    [Fact]
    public void ResolveOptionalModelName_WithSpaces_TrimsAndLowercases()
    {
        Assert.Equal("churn", CommandContext.ResolveOptionalModelName("  Churn  "));
    }
}
