using System.Linq;
using MLoop.Core.AutoML;
using Xunit;

namespace MLoop.Core.Tests.AutoML;

/// <summary>
/// Durable regression guard for upstream-007 stage 2 (tabular/DL assembly split): asserts
/// <c>MLoop.Core</c>'s compile-time referenced assemblies never re-include the Torch/Vision
/// deep-learning packages. If someone re-adds a
/// <c>&lt;PackageReference Include="Microsoft.ML.TorchSharp" /&gt;</c> (or Vision) to
/// <c>MLoop.Core.csproj</c> AND actually uses a type from it (so the reference survives
/// trimming from emitted metadata), this test fails.
/// </summary>
public class TabularBoundaryTests
{
    [Theory]
    [InlineData("Microsoft.ML.TorchSharp")]
    [InlineData("Microsoft.ML.Vision")]
    public void MLoopCore_does_not_reference_deep_learning_assemblies(string forbidden)
    {
        var referenced = typeof(AutoMLRunner).Assembly
            .GetReferencedAssemblies()
            .Select(a => a.Name)
            .ToList();
        Assert.DoesNotContain(forbidden, referenced);
    }
}
