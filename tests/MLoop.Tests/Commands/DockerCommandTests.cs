using MLoop.CLI.Commands;
using MLoop.Core.Storage;

namespace MLoop.Tests.Commands;

public class DockerCommandTests
{
    #region GenerateDockerfile

    [Fact]
    public void GenerateDockerfile_ContainsBaseImage()
    {
        var result = DockerCommand.GenerateDockerfile("default", 5000);

        Assert.Contains("mcr.microsoft.com/dotnet/aspnet", result);
        Assert.Contains("mcr.microsoft.com/dotnet/sdk", result);
    }

    [Fact]
    public void GenerateDockerfile_ContainsPort()
    {
        var result = DockerCommand.GenerateDockerfile("default", 8080);

        Assert.Contains("EXPOSE 8080", result);
        Assert.Contains("http://+:8080", result);
    }

    [Fact]
    public void GenerateDockerfile_ContainsModelName()
    {
        var result = DockerCommand.GenerateDockerfile("my-model", 5000);

        Assert.Contains("MLOOP_MODEL_NAME=my-model", result);
    }

    [Fact]
    public void GenerateDockerfile_ContainsMloopInstall()
    {
        var result = DockerCommand.GenerateDockerfile("default", 5000);

        Assert.Contains("dotnet tool install --global mloop", result);
    }

    [Fact]
    public void GenerateDockerfile_DoesNotUseUnquotedVersionWildcard()
    {
        // `--version *` is invalid: in a Dockerfile RUN the shell glob-expands the bare `*` to the
        // working-directory file list, breaking `docker build`. Omit it to install latest.
        var result = DockerCommand.GenerateDockerfile("default", 5000);

        Assert.DoesNotContain("--version *", result);
    }

    [Fact]
    public void GenerateDockerfile_ContainsHealthCheck()
    {
        var result = DockerCommand.GenerateDockerfile("default", 5000);

        Assert.Contains("HEALTHCHECK", result);
        // Port and path are the contract; the host spelling is not. The probe deliberately targets
        // the loopback literal rather than the name `localhost`, which resolves to ::1 first and is
        // refused on every probe because Kestrel binds IPv4 here.
        Assert.Contains(":5000/health", result);
    }

    [Fact]
    public void GenerateDockerfile_ContainsEntrypoint()
    {
        var result = DockerCommand.GenerateDockerfile("default", 5000);

        Assert.Contains("ENTRYPOINT", result);
        Assert.Contains("mloop", result);
        Assert.Contains("serve", result);
    }

    #endregion

    #region GenerateDockerignore

    [Fact]
    public void GenerateDockerignore_IgnoresTempFiles()
    {
        var result = DockerCommand.GenerateDockerignore();

        Assert.Contains(".mloop/temp/", result);
        Assert.Contains("*.tmp", result);
    }

    [Fact]
    public void GenerateDockerignore_IgnoresGitDirectory()
    {
        var result = DockerCommand.GenerateDockerignore();

        Assert.Contains(".git/", result);
    }

    [Fact]
    public void GenerateDockerignore_IgnoresIdeFiles()
    {
        var result = DockerCommand.GenerateDockerignore();

        Assert.Contains(".vs/", result);
        Assert.Contains(".vscode/", result);
        Assert.Contains(".idea/", result);
    }

    [Fact]
    public void GenerateDockerignore_KeepsModels()
    {
        var result = DockerCommand.GenerateDockerignore();

        // This asserted "!.mloop/models/" — a path no project has, so the exception kept nothing.
        // Models live in ExperimentLayout.ModelsDirectory, a sibling of .mloop/.
        Assert.Contains($"!{ExperimentLayout.ModelsDirectory}/", result);
    }

    #endregion

    #region GenerateDockerCompose

    [Fact]
    public void GenerateDockerCompose_ContainsServiceName()
    {
        var result = DockerCommand.GenerateDockerCompose("my-model", 5000);

        Assert.Contains("mloop-my-model", result);
    }

    [Fact]
    public void GenerateDockerCompose_ContainsPortMapping()
    {
        var result = DockerCommand.GenerateDockerCompose("default", 8080);

        Assert.Contains("8080:8080", result);
    }

    [Fact]
    public void GenerateDockerCompose_ContainsModelNameEnv()
    {
        var result = DockerCommand.GenerateDockerCompose("production", 5000);

        Assert.Contains("MLOOP_MODEL_NAME=production", result);
    }

    [Fact]
    public void GenerateDockerCompose_ContainsHealthcheck()
    {
        var result = DockerCommand.GenerateDockerCompose("default", 5000);

        Assert.Contains("healthcheck", result);
        Assert.Contains(":5000/health", result);
    }

    [Fact]
    public void GenerateDockerCompose_ContainsVolumeMount()
    {
        var result = DockerCommand.GenerateDockerCompose("default", 5000);

        // This asserted ".mloop/models", which is where no model has ever been — the mount pointed
        // at a directory docker would create empty, and the test pinned that as the contract.
        Assert.Contains($"./{ExperimentLayout.ModelsDirectory}:/app/{ExperimentLayout.ModelsDirectory}", result);
        Assert.Contains(":ro", result);
    }

    /// <summary>
    /// Serving <c>POST /embed</c> from the image is opt-in: the ONNX Runtime install line in the Dockerfile and the
    /// encoders' variable and mount in the compose file, commented. Uncommented as written, the compose file is still
    /// YAML and carries both into the service; the variable is the one <see cref="MLoop.Core.Embedding.EncoderStore"/> reads.
    /// </summary>
    [Fact]
    public void TheEmbeddingLines_AreOptIn_AndValidOnceUncommented()
    {
        var dockerfile = DockerCommand.GenerateDockerfile("default", 5000);
        Assert.Contains("# RUN mloop runtime install onnxruntime --quiet", dockerfile);

        var compose = DockerCommand.GenerateDockerCompose("default", 5000);
        var enabled = string.Join('\n', compose.Split('\n').Select(line =>
            line.TrimStart().StartsWith("# - ", StringComparison.Ordinal) && line.Contains("encoders", StringComparison.OrdinalIgnoreCase)
                ? line.Replace("# - ", "- ")
                : line));

        var service = (Dictionary<object, object>)((Dictionary<object, object>)
            new YamlDotNet.Serialization.Deserializer().Deserialize<Dictionary<object, object>>(enabled)["services"])["mloop-api"];
        Assert.Contains($"{MLoop.Core.Embedding.EncoderStore.RootVariable}=/encoders", ((List<object>)service["environment"]).Cast<string>());
        Assert.Contains("~/.mloop/encoders:/encoders:ro", ((List<object>)service["volumes"]).Cast<string>());
    }

    private static readonly MLoop.Core.Embedding.EncoderIdentity TextEncoder = new(
        "multilingual-minilm", "1", "abc", 384, Normalized: true, "xlm-roberta/1", MLoop.Core.Embedding.EncoderKind.Text, "def");

    /// <summary>
    /// A model trained on a text column through an encoder refuses to load without that encoder and ONNX Runtime
    /// (<see cref="MLoop.Core.Models.TextEncodings"/>), so for such a project the lines are on — not left commented for
    /// <c>POST /embed</c> — and the compose service carries the variable and the mount as YAML.
    /// </summary>
    [Fact]
    public void ModelsThatReadTextThroughAnEncoder_GetTheRuntimeAndTheMount()
    {
        var dockerfile = DockerCommand.GenerateDockerfile("default", 5000, [TextEncoder]);
        Assert.Contains("\nRUN mloop runtime install onnxruntime --quiet", dockerfile);
        Assert.DoesNotContain("# RUN mloop runtime install", dockerfile);
        Assert.Contains("multilingual-minilm", dockerfile);
        Assert.Contains($"ENV {MLoop.Core.Embedding.EncoderStore.RootVariable}=/encoders", dockerfile);

        var service = (Dictionary<object, object>)((Dictionary<object, object>)
            new YamlDotNet.Serialization.Deserializer().Deserialize<Dictionary<object, object>>(
                DockerCommand.GenerateDockerCompose("default", 5000, [TextEncoder]))["services"])["mloop-api"];
        Assert.Contains($"{MLoop.Core.Embedding.EncoderStore.RootVariable}=/encoders", ((List<object>)service["environment"]).Cast<string>());
        Assert.Contains("~/.mloop/encoders:/encoders:ro", ((List<object>)service["volumes"]).Cast<string>());
    }

    [Fact]
    public void TextEncodersUnder_ReadsEveryModelsEncodings_OncePerId()
    {
        var root = Path.Combine(Path.GetTempPath(), "mloop-docker-" + Guid.NewGuid().ToString("N"));
        try
        {
            var models = Path.Combine(root, ExperimentLayout.ModelsDirectory, "default");
            foreach (var dir in new[] { Path.Combine(models, ExperimentLayout.ProductionDirectory), Path.Combine(models, ExperimentLayout.StagingDirectory, "exp-002") })
            {
                Directory.CreateDirectory(dir);
                new MLoop.Core.Models.TextEncodings([new("review", TextEncoder), new("title", TextEncoder)])
                    .WriteBeside(Path.Combine(dir, ExperimentLayout.ModelFileName));
            }
            Directory.CreateDirectory(Path.Combine(models, ExperimentLayout.StagingDirectory, "exp-001"));

            var found = DockerCommand.TextEncodersUnder(root);

            Assert.Equal("multilingual-minilm", Assert.Single(found).Id);
            Assert.Empty(DockerCommand.TextEncodersUnder(Path.Combine(root, "nowhere")));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void GenerateDockerCompose_ContainsRestartPolicy()
    {
        var result = DockerCommand.GenerateDockerCompose("default", 5000);

        Assert.Contains("restart: unless-stopped", result);
    }

    #endregion
}
