namespace MLoop.Tests.Documentation;

/// <summary>
/// Locating and reading this repository's sources — the part every source-text guard needs and none
/// of them should own a copy of.
/// </summary>
/// <remarks>
/// The guards in this namespace assert rules about the test tree itself: which dispatch a test may
/// drive, which documented examples must parse, which assertions may bind a wall clock, which
/// statics an assembly may touch. Each rule is its own; what they share is finding the repository
/// and reading its files, and four copies of that is how one of them ends up walking a slightly
/// different tree than the others. The matching stays with each guard, deliberately — a common
/// "rule engine" would hide the one thing each guard exists to state.
/// </remarks>
internal static class RepoSourceTree
{
    /// <summary>The repository root, found by walking up from the build output.</summary>
    internal static string RepoRoot { get; } = FindRepoRoot();

    /// <summary>The <c>tests/</c> directory.</summary>
    internal static string TestsRoot { get; } = Path.Combine(RepoRoot, "tests");

    /// <summary>
    /// Every C# source file under the given repository-relative directories, excluding build output.
    /// </summary>
    /// <remarks>
    /// A guard about what the product does — rather than about how it is tested — reads these.
    /// </remarks>
    internal static IEnumerable<string> ProductionSourceFiles(params string[] directories) =>
        directories
            .Select(d => Path.Combine(RepoRoot, d))
            .Where(Directory.Exists)
            .SelectMany(d => Directory.EnumerateFiles(d, "*.cs", SearchOption.AllDirectories))
            .Where(NotBuildOutput);

    /// <summary>
    /// Every C# source file under <see cref="TestsRoot"/>, excluding build output and the caller
    /// itself — a guard that names the pattern it forbids would otherwise always find one.
    /// </summary>
    internal static IEnumerable<string> SourceFiles(string? excludingFileNamed = null) =>
        Directory.EnumerateFiles(TestsRoot, "*.cs", SearchOption.AllDirectories)
            .Where(f => NotBuildOutput(f)
                     && (excludingFileNamed is null
                         || !Path.GetFileName(f).Equals(excludingFileNamed + ".cs", StringComparison.Ordinal)));

    private static bool NotBuildOutput(string file) =>
        !file.Contains(Path.Combine("bin", ""), StringComparison.Ordinal)
        && !file.Contains(Path.Combine("obj", ""), StringComparison.Ordinal);

    /// <summary>The path of <paramref name="file"/> relative to <see cref="TestsRoot"/>.</summary>
    internal static string Relative(string file) => Path.GetRelativePath(TestsRoot, file);

    /// <summary>
    /// The test assembly a source file belongs to — the first path segment under <c>tests/</c>.
    /// </summary>
    internal static string AssemblyOf(string file) =>
        Relative(file).Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)[0];

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "MLoop.slnx")))
            dir = dir.Parent;

        if (dir == null)
            throw new InvalidOperationException(
                $"Could not locate MLoop.slnx by walking up from {AppContext.BaseDirectory} — " +
                $"{nameof(RepoSourceTree)} assumes it runs from within the repo's build output tree.");

        return dir.FullName;
    }
}
