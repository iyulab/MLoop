using MLoop.Core.Training;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// MLoop.Core is a published library, and it searches with its own trainer search: its compiled assembly
/// references no ML.NET AutoML. Read from the compiled references, so a type that slips back in through a
/// transitive package is seen even when no source line names the namespace.
/// </summary>
public class AutoMLDependencyTests
{
    private const string AutoMLAssembly = "Microsoft.ML.AutoML";

    [Fact]
    public void MLoopCore_DoesNotReferenceAutoML()
    {
        Assert.DoesNotContain(typeof(TrainingRunner).Assembly.GetReferencedAssemblies(), a => a.Name == AutoMLAssembly);
    }

    // The check must be able to see a reference: the same read finds one MLoop.Core does keep. (This test
    // assembly kept AutoML itself until the tuner's upstream trace became a fixture.)
    [Fact]
    public void TheCheck_SeesAReferenceThatIsThere()
    {
        Assert.Contains(typeof(TrainingRunner).Assembly.GetReferencedAssemblies(), a => a.Name == "Microsoft.ML.Data");
    }

    [Fact]
    public void NoTestAssemblyReferencesAutoMLEither()
    {
        Assert.DoesNotContain(typeof(AutoMLDependencyTests).Assembly.GetReferencedAssemblies(), a => a.Name == AutoMLAssembly);
    }
}
