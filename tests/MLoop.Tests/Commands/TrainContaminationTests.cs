using MLoop.CLI.Infrastructure.Configuration;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>contamination</c> under a model in <c>mloop.yaml</c> — the share of its training rows a table anomaly model calls
/// anomalous. A value that is no share of rows, or one on a task that has no such threshold, is refused instead of silently
/// meaning nothing.
/// </summary>
public sealed class TrainContaminationTests
{
    private static ModelDefinition Definition(string task, double? contamination)
        => new() { Task = task, Label = "y", Contamination = contamination };

    [Theory]
    [InlineData(0.01)]
    [InlineData(0.05)]
    [InlineData(0.49)]
    public void AShareOnATableAnomalyModel_HasNoConflict_AndReachesTheTrainingConfig(double contamination)
    {
        var definition = Definition("anomaly-detection", contamination);

        Assert.Null(ModelTrainingConfig.ContaminationConflict(definition));
        Assert.Equal(contamination, ModelTrainingConfig.From(definition, "default", "table.csv", 60, 0.2).Contamination);
    }

    [Fact]
    public void NoContamination_LeavesTheDefault()
    {
        var definition = Definition("anomaly-detection", null);

        Assert.Null(ModelTrainingConfig.ContaminationConflict(definition));
        Assert.Null(ModelTrainingConfig.From(definition, "default", "table.csv", 60, 0.2).Contamination);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.5)]
    [InlineData(1.0)]
    [InlineData(-0.1)]
    [InlineData(double.NaN)]
    public void AValueThatIsNoShareOfRows_IsRefused(double contamination)
        => Assert.Contains("not a share of rows", ModelTrainingConfig.ContaminationConflict(Definition("anomaly-detection", contamination)), StringComparison.Ordinal);

    [Theory]
    [InlineData("regression")]
    [InlineData("time-series-anomaly")]
    [InlineData("clustering")]
    public void AContaminationOnAnotherTask_IsRefusedAndNamesTheTask(string task)
    {
        var message = ModelTrainingConfig.ContaminationConflict(Definition(task, 0.1));

        Assert.NotNull(message);
        Assert.Contains($"'{task}'", message, StringComparison.Ordinal);
        Assert.Contains("anomaly-detection", message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheYamlKey_IsReadFromTheFile_AndTheAuditKnowsIt()
    {
        const string Yaml = "models:\n  default:\n    task: anomaly-detection\n    label: y\n    contamination: 0.1\n";

        Assert.Empty(ConfigKeyAudit.Find<MLoopConfig>(Yaml));
        Assert.Single(ConfigKeyAudit.Find<MLoopConfig>(Yaml.Replace("contamination", "contamnation")));
    }

    [Fact]
    public void ThePartnersOfTheKey_ReachTheEffectiveDefinition()
    {
        var config = new MLoopConfig { Models = new Dictionary<string, ModelDefinition> { ["default"] = Definition("anomaly-detection", 0.1) } };

        Assert.Equal(0.1, new ConfigMerger().GetEffectiveModelDefinition(config, "default").Contamination);
    }
}
