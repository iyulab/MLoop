using MLoop.Core.Storage;

namespace MLoop.Core.Tests.Storage;

/// <summary>
/// Which model a request means. The rule used to live in the CLI, and the CLI is not what a
/// generated container runs — so the variable `mloop docker` writes into the image was read by
/// nothing in the process that serves the requests.
/// </summary>
/// <remarks>
/// These tests set a process-wide environment variable, so the assembly's parallelization is
/// already disabled (see the collection-behaviour contract test); each test restores what it found.
/// </remarks>
public class ModelNameResolveTests : IDisposable
{
    private readonly string? _saved = Environment.GetEnvironmentVariable(ModelName.EnvironmentVariable);

    private static void Set(string? value)
        => Environment.SetEnvironmentVariable(ModelName.EnvironmentVariable, value);

    public void Dispose()
    {
        Set(_saved);
        GC.SuppressFinalize(this);
    }

    [Fact]
    public void ANamedModelWins()
    {
        Set("churn");

        Assert.Equal("revenue", ModelName.Resolve("revenue"));
    }

    /// <summary>The case the variable exists for: a request that names no model.</summary>
    [Fact]
    public void TheEnvironmentNamesTheModelWhenTheRequestDoesNot()
    {
        Set("churn");

        Assert.Equal("churn", ModelName.Resolve(null));
        Assert.Equal("churn", ModelName.Resolve(""));
        Assert.Equal("churn", ModelName.Resolve("   "));
    }

    [Fact]
    public void TheDefaultIsTheLastResort()
    {
        Set(null);

        Assert.Equal(ModelName.Default, ModelName.Resolve(null));
        Assert.Equal("default", ModelName.Default);
    }

    /// <summary>A variable set to nothing is not a name.</summary>
    [Fact]
    public void ABlankEnvironmentValueIsIgnored()
    {
        Set("   ");

        Assert.Equal(ModelName.Default, ModelName.Resolve(null));
    }

    [Fact]
    public void BothSourcesAreNormalized()
    {
        Set("  Churn  ");
        Assert.Equal("churn", ModelName.Resolve(null));

        Assert.Equal("revenue", ModelName.Resolve("  Revenue  "));
    }

    /// <summary>
    /// The variable name is the one the generated Dockerfile writes. If these ever diverge the
    /// image goes back to serving the wrong model, silently, which is how this started.
    /// </summary>
    [Fact]
    public void TheVariableIsNamedWhatTheGeneratedImageSets()
        => Assert.Equal("MLOOP_MODEL_NAME", ModelName.EnvironmentVariable);
}
