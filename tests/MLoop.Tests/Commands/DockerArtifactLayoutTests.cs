using MLoop.CLI.Commands;
using MLoop.Core.Storage;

namespace MLoop.Tests.Commands;

/// <summary>
/// What <c>mloop docker</c> generates has to point at the directory a project actually keeps its
/// models in.
/// </summary>
/// <remarks>
/// <para>
/// It pointed at <c>.mloop/models</c> — a path no project has. Models live in
/// <see cref="ExperimentLayout.ModelsDirectory"/>, a sibling of <c>.mloop/</c>, so the generated
/// image copied the project marker and the configuration and left the model behind, and the compose
/// file mounted a directory that does not exist. The container starts, answers its health check,
/// and has nothing to serve — a failure that looks like a working deployment until the first
/// prediction request.
/// </para>
/// <para>
/// Found by reading the generated files rather than by building them: the daemon was not available,
/// and this half of the check never needed it. The generator now consumes the layout authority
/// instead of restating the path, which is what these tests pin.
/// </para>
/// </remarks>
public class DockerArtifactLayoutTests
{
    private const string StaleModelsPath = ".mloop/models";

    [Fact]
    public void TheImageCopiesTheModelsDirectory()
    {
        var dockerfile = DockerCommand.GenerateDockerfile("default", 5000);

        Assert.Contains($"COPY {ExperimentLayout.ModelsDirectory}/", dockerfile);
        Assert.DoesNotContain(StaleModelsPath, dockerfile);
        // What matters is that the models arrive at the authority's path. Which stage they come
        // *through* is not the contract: the build stage no longer holds the project at all, since
        // it exists only to acquire the tool, so the models are copied straight from the context
        //.
    }

    [Fact]
    public void TheComposeFileMountsTheModelsDirectory()
    {
        var compose = DockerCommand.GenerateDockerCompose("default", 5000);

        Assert.Contains($"./{ExperimentLayout.ModelsDirectory}:/app/{ExperimentLayout.ModelsDirectory}", compose);
        Assert.DoesNotContain(StaleModelsPath, compose);
    }

    /// <summary>
    /// And the ignore file does not exclude what the image is meant to carry.
    /// </summary>
    [Fact]
    public void TheIgnoreFileKeepsTheModelsDirectory()
    {
        var dockerignore = DockerCommand.GenerateDockerignore();

        Assert.Contains($"!{ExperimentLayout.ModelsDirectory}/", dockerignore);
        Assert.DoesNotContain(StaleModelsPath, dockerignore);
    }

    /// <summary>
    /// The check can see what it forbids: had the generator kept the old path, every assertion above
    /// would have to fail rather than pass on a technicality of substring matching.
    /// </summary>
    [Fact]
    public void TheCheckRecognisesTheShapeItForbids()
    {
        const string stale = "COPY --from=build /src/.mloop/models ./.mloop/models";

        Assert.Contains(StaleModelsPath, stale);
        Assert.DoesNotContain($"COPY {ExperimentLayout.ModelsDirectory}/", stale);
    }
}
