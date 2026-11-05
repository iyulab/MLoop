using System.Reflection;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.Core.Preprocessing;

namespace MLoop.Tests.Configuration;

/// <summary>
/// A setting a model has in <c>mloop.yaml</c> reaches the training that reads it.
/// </summary>
/// <remarks>
/// <see cref="ConfigMerger"/> builds the effective <see cref="ModelDefinition"/> by copying it field by field. A field added to
/// the definition and left out of the copy is lost without a message: the file says <c>ensemble: true</c>, the run trains a
/// single model (a second copy, for two files defining one model, had the same gap and was never called — it is gone). The copy
/// is an explicit list on purpose (a field may merge differently), so this holds it to the properties the type has — it fills
/// every one, sends the definition through and asks which came out empty.
/// </remarks>
public class ModelDefinitionCopySeamTests
{
    private static readonly PropertyInfo[] Settings = typeof(ModelDefinition)
        .GetProperties(BindingFlags.Public | BindingFlags.Instance)
        .Where(p => p.CanWrite)
        .ToArray();

    /// <summary>A definition with every setting set to something other than its default.</summary>
    private static ModelDefinition Full()
    {
        var definition = new ModelDefinition { Task = "regression", Label = "y" };
        foreach (var setting in Settings.Where(p => p.Name is not (nameof(ModelDefinition.Task) or nameof(ModelDefinition.Label))))
            setting.SetValue(definition, ValueFor(setting.PropertyType, setting.Name));
        return definition;
    }

    private static object ValueFor(Type type, string name)
    {
        type = Nullable.GetUnderlyingType(type) ?? type;
        if (type == typeof(string)) return name;
        if (type == typeof(int)) return 7;
        if (type == typeof(double)) return 0.4;
        if (type == typeof(bool)) return true;
        if (type == typeof(TrainingSettings)) return new TrainingSettings { MaxTrials = 3 };
        if (type == typeof(GatePolicy)) return new GatePolicy { NoWorseThanProduction = true };
        if (type == typeof(List<PrepStep>)) return new List<PrepStep> { new() { Type = "fill-missing" } };
        if (type == typeof(Dictionary<string, ColumnOverride>)) return new Dictionary<string, ColumnOverride> { ["a"] = new() { Type = "text" } };
        throw new NotSupportedException($"ModelDefinition.{name} is a {type.Name}: teach ValueFor to fill it, or the copies' guard cannot see it.");
    }

    private static IEnumerable<string> Empty(ModelDefinition definition)
        => Settings.Where(p => p.GetValue(definition) is null).Select(p => p.Name);

    [Fact]
    public void TheGuard_FillsEverySetting_SoAnEmptyOneMeansALostCopy()
    {
        var full = Full();

        Assert.Empty(Empty(full));
        Assert.True(Settings.Length >= 18, $"the scan saw {Settings.Length} settings of a model — it should see the whole type");
    }

    [Fact]
    public void AnEffectiveDefinition_KeepsEverySettingOfTheFile()
    {
        var config = new MLoopConfig { Models = new Dictionary<string, ModelDefinition> { ["default"] = Full() } };

        var effective = new ConfigMerger().GetEffectiveModelDefinition(config, "default");

        Assert.Empty(Empty(effective));
    }
}
