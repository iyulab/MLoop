using MLoop.CLI.Commands;

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
    public void GenerateDockerfile_ContainsHealthCheck()
    {
        var result = DockerCommand.GenerateDockerfile("default", 5000);

        Assert.Contains("HEALTHCHECK", result);
        Assert.Contains("http://localhost:5000/health", result);
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

        Assert.Contains("!.mloop/models/", result);
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
        Assert.Contains("http://localhost:5000/health", result);
    }

    [Fact]
    public void GenerateDockerCompose_ContainsVolumeMount()
    {
        var result = DockerCommand.GenerateDockerCompose("default", 5000);

        Assert.Contains(".mloop/models", result);
        Assert.Contains(":ro", result);
    }

    [Fact]
    public void GenerateDockerCompose_ContainsRestartPolicy()
    {
        var result = DockerCommand.GenerateDockerCompose("default", 5000);

        Assert.Contains("restart: unless-stopped", result);
    }

    #endregion
}
