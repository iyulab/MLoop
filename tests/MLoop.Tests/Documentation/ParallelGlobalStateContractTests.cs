using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// A test assembly that touches a process-wide static must not run its classes in parallel.
/// </summary>
/// <remarks>
/// <para>
/// <c>Console.Out</c>, <c>AnsiConsole.Console</c>, the current directory and the deep-learning
/// registry are one per process, not one per test. A test that points one of them somewhere else
/// while an unrelated class is running has changed that class's world mid-assertion — and the
/// failure surfaces as a wrong value in whichever test happened to be looking, which is why it
/// reads as flakiness rather than as a race.
/// </para>
/// <para>
/// <c>[Collection]</c> does not fix this and is the tempting answer: it orders the classes that
/// share a name and says nothing about the rest of the assembly. Assembly-wide serialization is
/// what actually closes it. This asserts the pairing — touching the statics, and declaring the
/// serialization — so a new test that reaches for one of them in a parallelizing assembly fails
/// here, at the moment it is added, rather than three months later as an unreproducible failure in
/// something it never mentioned.
/// </para>
/// </remarks>
public class ParallelGlobalStateContractTests
{
    /// <summary>Mutations of state that is one-per-process.</summary>
    private static readonly (string Token, string What)[] ProcessGlobalMutations =
    [
        ("Console.SetOut", "redirects the process's standard output"),
        ("Console.SetError", "redirects the process's standard error"),
        ("Console.SetIn", "redirects the process's standard input"),
        ("AnsiConsole.Console =", "replaces the ambient Spectre console"),
        ("Directory.SetCurrentDirectory", "moves the process's working directory"),
        ("DeepLearningRegistry.Register", "mutates a registry every loader resolution reads"),
    ];

    private static readonly Regex SerializationDeclared = new(
        @"\[\s*assembly\s*:\s*CollectionBehavior\s*\(\s*DisableTestParallelization\s*=\s*true",
        RegexOptions.Compiled);

    [Fact]
    public void AnAssemblyThatTouchesProcessGlobalStateIsSerialized()
    {
        var files = TestSourceTree.SourceFiles(
            excludingFileNamed: nameof(ParallelGlobalStateContractTests)).ToList();

        var serialized = files
            .Where(f => SerializationDeclared.IsMatch(File.ReadAllText(f)))
            .Select(TestSourceTree.AssemblyOf)
            .ToHashSet(StringComparer.Ordinal);

        var offenders =
            (from file in files
             where !serialized.Contains(TestSourceTree.AssemblyOf(file))
             from line in File.ReadAllLines(file).Select((Text, Number) => (Text, Number))
             from mutation in ProcessGlobalMutations
             where line.Text.Contains(mutation.Token, StringComparison.Ordinal)
             select $"{TestSourceTree.Relative(file)}:{line.Number + 1} — "
                  + $"{mutation.Token} {mutation.What}")
            .ToList();

        Assert.True(
            offenders.Count == 0,
            "These run in an assembly whose classes execute in parallel, so the static they move "
            + "is moved out from under whatever else is running. Add "
            + "[assembly: CollectionBehavior(DisableTestParallelization = true)] to that assembly "
            + "— a [Collection] attribute orders only the classes that share it:"
            + Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }

    [Fact]
    public void TheCheckKnowsWhichAssembliesAreSerialized()
    {
        // Negative control for the exemption itself: if the declaration were never recognised, the
        // assertion above would pass vacuously by finding no serialized assembly and no offender.
        var serialized = TestSourceTree.SourceFiles()
            .Where(f => SerializationDeclared.IsMatch(File.ReadAllText(f)))
            .Select(TestSourceTree.AssemblyOf)
            .ToHashSet(StringComparer.Ordinal);

        Assert.Contains("MLoop.Tests", serialized);
        Assert.Contains("MLoop.Core.Tests", serialized);
    }
}
