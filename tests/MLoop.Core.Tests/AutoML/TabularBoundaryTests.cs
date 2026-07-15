using System;
using System.IO;
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

    /// <summary>
    /// Complementary guard with different failing power than the reflection-based test above:
    /// that test only catches a re-added Torch/Vision PackageReference if a TYPE from it is
    /// actually used (surviving trimming from emitted assembly metadata). This test reads
    /// <c>MLoop.Core.csproj</c>'s raw text and catches a dangling, wholly UNUSED
    /// <c>&lt;PackageReference&gt;</c> too — one that never appears in
    /// <see cref="System.Reflection.Assembly.GetReferencedAssemblies"/> because no code
    /// references it, but that still drags Torch/Vision DLLs into MLoop.Core's publish output.
    /// </summary>
    [Fact]
    public void MLoopCoreCsproj_does_not_contain_deep_learning_package_references()
    {
        var csprojPath = FindMLoopCoreCsproj();
        Assert.True(File.Exists(csprojPath),
            $"Could not locate src/MLoop.Core/MLoop.Core.csproj by walking up from " +
            $"'{AppContext.BaseDirectory}'. This test must find and read the real csproj, " +
            "not vacuously pass.");

        var text = File.ReadAllText(csprojPath);

        Assert.DoesNotContain("Include=\"Microsoft.ML.TorchSharp\"", text);
        Assert.DoesNotContain("Include=\"Microsoft.ML.Vision\"", text);
    }

    private static string? FindMLoopCoreCsproj()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, "src", "MLoop.Core", "MLoop.Core.csproj");
            if (File.Exists(candidate))
                return candidate;

            dir = dir.Parent;
        }

        return null;
    }
}
