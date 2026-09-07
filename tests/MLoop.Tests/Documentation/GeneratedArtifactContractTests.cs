using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// Every environment variable MLoop writes into the deployment files it generates is read somewhere.
/// </summary>
/// <remarks>
/// <para>
/// <c>mloop docker</c> writes a Dockerfile and a compose file for the user, and <c>mloop serve</c>
/// sets variables before starting the API. Those are promises: a reader who sees
/// <c>ENV MLOOP_PROJECT_ROOT=/app</c> concludes that moving the image's working directory is safe,
/// and one who sees <c>MLOOP_MODEL_NAME</c> concludes the image serves that model.
/// </para>
/// <para>
/// Both promises were false for months. The variables were written and read by nothing: the images
/// worked because their working directory happened to be the project and the model happened to be
/// named <c>default</c>. Nothing failed, so nothing said so. The generated file is a contract with
/// a user who cannot see this repository, and it is only true if something here honours it.
/// </para>
/// </remarks>
public class GeneratedArtifactContractTests
{
    /// <summary>Where MLoop writes variables into artifacts or a child process, rather than reading them.</summary>
    private static readonly string[] EmitterFiles = ["DockerCommand.cs", "ServeCommand.cs"];

    private static readonly Regex MLoopVariable = new(@"\bMLOOP_[A-Z0-9_]+\b", RegexOptions.Compiled);

    [Fact]
    public void EveryVariableTheGeneratedDeploymentSetsIsReadSomewhere()
    {
        var production = RepoSourceTree.ProductionSourceFiles("src", "tools").ToList();

        var emitted = production
            .Where(f => EmitterFiles.Contains(Path.GetFileName(f), StringComparer.Ordinal))
            .SelectMany(f => MLoopVariable.Matches(File.ReadAllText(f)).Select(m => m.Value))
            .ToHashSet(StringComparer.Ordinal);

        Assert.NotEmpty(emitted); // the generated files still set variables at all

        var readSomewhereElse = production
            .Where(f => !EmitterFiles.Contains(Path.GetFileName(f), StringComparer.Ordinal))
            .SelectMany(f => MLoopVariable.Matches(File.ReadAllText(f)).Select(m => m.Value))
            .ToHashSet(StringComparer.Ordinal);

        var decoration = emitted.Except(readSomewhereElse).Order(StringComparer.Ordinal).ToList();

        Assert.True(
            decoration.Count == 0,
            "These are written into the deployment files MLoop generates (or set for the API "
            + "process) and read by nothing outside those two files, so the generated file promises "
            + "a user something the product does not do. Read them where the answer is decided, or "
            + "stop writing them:" + Environment.NewLine + string.Join(Environment.NewLine, decoration));
    }

    [Fact]
    public void TheCheckSeesTheVariablesItIsAbout()
    {
        // Negative control for the scan itself: if the emitter files stopped matching — renamed,
        // moved, or the pattern narrowed — the assertion above would pass by finding nothing emitted.
        var emitted = RepoSourceTree.ProductionSourceFiles("src", "tools")
            .Where(f => EmitterFiles.Contains(Path.GetFileName(f), StringComparer.Ordinal))
            .SelectMany(f => MLoopVariable.Matches(File.ReadAllText(f)).Select(m => m.Value))
            .ToHashSet(StringComparer.Ordinal);

        Assert.Contains("MLOOP_PROJECT_ROOT", emitted);
        Assert.Contains("MLOOP_MODEL_NAME", emitted);
    }
}
