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

    /// <summary>
    /// The variables one file writes. A name can reach the generated artifact two ways: spelled as
    /// a literal, or interpolated from the constant that owns it — which is what
    /// <c>MLOOP_MODEL_NAME</c> now does, so the emitter and the process that reads it cannot
    /// disagree about the spelling. Both count as emitting it; counting only literals would let
    /// this report a file as emitting nothing the moment it started doing the right thing.
    /// </summary>
    private static IEnumerable<string> VariablesIn(string file)
    {
        var source = File.ReadAllText(file);

        foreach (Match m in MLoopVariable.Matches(source))
            yield return m.Value;

        if (source.Contains("ModelName.EnvironmentVariable", StringComparison.Ordinal))
            yield return MLoop.Core.Storage.ModelName.EnvironmentVariable;
    }

    [Fact]
    public void EveryVariableTheGeneratedDeploymentSetsIsReadSomewhere()
    {
        var production = RepoSourceTree.ProductionSourceFiles("src", "tools").ToList();

        var emitted = production
            .Where(f => EmitterFiles.Contains(Path.GetFileName(f), StringComparer.Ordinal))
            .SelectMany(VariablesIn)
            .ToHashSet(StringComparer.Ordinal);

        Assert.NotEmpty(emitted); // the generated files still set variables at all

        var readSomewhereElse = production
            .Where(f => !EmitterFiles.Contains(Path.GetFileName(f), StringComparer.Ordinal))
            .SelectMany(VariablesIn)
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
            .SelectMany(VariablesIn)
            .ToHashSet(StringComparer.Ordinal);

        Assert.Contains("MLOOP_PROJECT_ROOT", emitted);
        Assert.Contains("MLOOP_MODEL_NAME", emitted);
    }

    /// <summary>
    /// The variable the generated image sets is the one the serving process reads. It was not:
    /// `mloop docker` wrote MLOOP_MODEL_NAME, the CLI's resolver read it, and the image runs the
    /// API — which resolved the model name in eight places of its own, none of which looked at the
    /// environment. An image built for one model served `default`.
    /// </summary>
    [Fact]
    public void TheGeneratedImageSetsTheVariableTheServerReads()
    {
        var emitted = MLoop.CLI.Commands.DockerCommand.GenerateDockerfile("churn", 8080);

        Assert.Contains($"ENV {MLoop.Core.Storage.ModelName.EnvironmentVariable}=churn", emitted);
    }

    /// <summary>
    /// And nothing in the serving process resolves the name on its own again. A scan, because the
    /// failure being prevented is a ninth copy appearing, which no amount of correct behaviour in
    /// the other eight would reveal.
    /// </summary>
    [Fact]
    public void TheApiDoesNotResolveAModelNameItself()
    {
        var scanned = RepoSourceTree.ProductionSourceFiles(Path.Combine("tools", "MLoop.API")).ToList();
        Assert.Contains(scanned, f => Path.GetFileName(f) == "Program.cs");

        var shape = new Regex(
            @"IsNullOrWhiteSpace[^;]{0,120}DefaultModelName", RegexOptions.Compiled | RegexOptions.Singleline);

        var offenders = scanned
            .Where(f => shape.IsMatch(File.ReadAllText(f)))
            .Select(RepoSourceTree.RelativeToRoot)
            .Order(StringComparer.Ordinal)
            .ToList();

        // The scan can see the shape it forbids.
        Assert.Matches(shape,
            "var m = string.IsNullOrWhiteSpace(name) ? ConfigDefaults.DefaultModelName : name.Trim();");
        Assert.DoesNotMatch(shape, "var m = MLoop.Core.Storage.ModelName.Resolve(name);");

        Assert.True(offenders.Count == 0,
            "Which model a request means is decided by " + nameof(MLoop.Core.Storage.ModelName)
            + ".Resolve, which is also where the environment variable a generated image sets is "
            + "read:\n"
            + string.Join('\n', offenders));
    }
}
