using System.Reflection;
using MLoop.Search;

namespace MLoop.Search.Tests;

/// <summary>
/// MLoop.Search knows ML.NET and nothing else of MLoop: MLoop.Core depends on it, never the
/// reverse, and Microsoft.ML.AutoML — the search this assembly replaces — is not a dependency.
/// Checked on the compiled assembly's references, not on source text.
/// </summary>
public class DependencyBoundaryTests
{
    [Fact]
    public void TheSearchAssembly_ReferencesOnlyTheRuntimeAndMLNet()
    {
        var references = typeof(ColumnRoles).Assembly.GetReferencedAssemblies().Select(r => r.Name!);

        Assert.Empty(Violations(references));
    }

    // The check must be able to see what it forbids, and must not refuse what it allows.

    [Theory]
    [InlineData("MLoop.Core")]
    [InlineData("MLoop.Extensibility")]
    [InlineData("Microsoft.ML.AutoML")]
    [InlineData("Spectre.Console")]
    public void TheCheck_RefusesAForbiddenReference(string forbidden)
    {
        Assert.Equal([forbidden], Violations(["System.Runtime", forbidden]));
    }

    [Theory]
    [InlineData("System.Runtime")]
    [InlineData("System.Collections")]
    [InlineData("netstandard")]
    [InlineData("Microsoft.ML.Core")]
    [InlineData("Microsoft.ML.FastTree")]
    public void TheCheck_AllowsTheRuntimeAndMLNet(string allowed)
    {
        Assert.Empty(Violations([allowed]));
    }

    private static List<string> Violations(IEnumerable<string> references) =>
        [.. references.Where(name => !IsAllowed(name))];

    private static bool IsAllowed(string name) =>
        name == "netstandard"
        || name == "mscorlib"
        || name.StartsWith("System", StringComparison.Ordinal)
        || name.StartsWith("Microsoft.Win32", StringComparison.Ordinal)
        || (name.StartsWith("Microsoft.ML", StringComparison.Ordinal) && name != "Microsoft.ML.AutoML");
}
