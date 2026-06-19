using MLoop.Core.Scripting;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Core.Tests.Scripting;

/// <summary>
/// Compiles the shipped preprocessing example scripts (examples/preprocessing-scripts/*.cs)
/// through the real <see cref="ScriptLoader"/>.
///
/// Rationale: those files live outside any csproj, so the build never compiles them. They drifted
/// from the authoritative IPreprocessingScript contract (wrong namespace, non-existent
/// PreprocessingResult/PreprocessingContext types, GetTempPath) and shipped broken. This test makes
/// the examples first-class compile targets so copy-paste consumers always get working code.
/// </summary>
public class ExampleScriptsCompilationTests : IDisposable
{
    private readonly string _cacheDirectory;

    public ExampleScriptsCompilationTests()
    {
        _cacheDirectory = Path.Combine(Path.GetTempPath(), "mloop_examples_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_cacheDirectory);
    }

    public void Dispose()
    {
        if (Directory.Exists(_cacheDirectory))
        {
            Directory.Delete(_cacheDirectory, recursive: true);
        }
    }

    public static IEnumerable<object[]> ExampleScriptFiles()
    {
        var dir = FindExampleDirectory();
        foreach (var file in Directory.GetFiles(dir, "*.cs").OrderBy(f => f))
        {
            yield return new object[] { Path.GetFileName(file) };
        }
    }

    [Theory]
    [MemberData(nameof(ExampleScriptFiles))]
    public async Task ExampleScript_CompilesAndLoadsAsIPreprocessingScript(string fileName)
    {
        var scriptPath = Path.Combine(FindExampleDirectory(), fileName);
        var loader = new ScriptLoader(Path.Combine(_cacheDirectory, Path.GetFileNameWithoutExtension(fileName)));

        var instances = await loader.LoadScriptAsync<IPreprocessingScript>(scriptPath);

        // Every shipped example must compile and expose exactly one IPreprocessingScript.
        Assert.Single(instances);
        Assert.IsAssignableFrom<IPreprocessingScript>(instances[0]);
    }

    /// <summary>
    /// Walks up from the test output directory to locate examples/preprocessing-scripts in the repo.
    /// </summary>
    private static string FindExampleDirectory()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, "examples", "preprocessing-scripts");
            if (Directory.Exists(candidate))
            {
                return candidate;
            }
            dir = dir.Parent;
        }

        throw new DirectoryNotFoundException(
            "Could not locate examples/preprocessing-scripts from " + AppContext.BaseDirectory);
    }
}
