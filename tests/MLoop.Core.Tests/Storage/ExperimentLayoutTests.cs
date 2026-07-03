using MLoop.Core.Storage;

namespace MLoop.Core.Tests.Storage;

/// <summary>
/// Pins the canonical experiment-staging layout values. These constants are the single authority
/// shared by MLoop.CLI's ExperimentStore (writer) and MLoop.Ops services (readers); F-32/F-33 were
/// caused by an independent Ops copy drifting to a non-existent "experiments" directory. Locking the
/// values here makes any future drift a failing test rather than a silently broken read path.
/// </summary>
public class ExperimentLayoutTests
{
    [Fact]
    public void CanonicalValues_AreLocked()
    {
        Assert.Equal("models", ExperimentLayout.ModelsDirectory);
        Assert.Equal("staging", ExperimentLayout.StagingDirectory);
        Assert.Equal("production", ExperimentLayout.ProductionDirectory);
        Assert.Equal("model.zip", ExperimentLayout.ModelFileName);
        Assert.Equal("residual-model.zip", ExperimentLayout.ResidualModelFileName);
        Assert.Equal("metadata.json", ExperimentLayout.MetadataFileName);
        Assert.Equal("metrics.json", ExperimentLayout.MetricsFileName);
        Assert.Equal("config.json", ExperimentLayout.ConfigFileName);
        Assert.Equal("experiment-index.json", ExperimentLayout.IndexFileName);
        Assert.Equal("registry.json", ExperimentLayout.RegistryFileName);
    }
}
