using System.CommandLine;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Storage;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// Docker command to generate containerization files
/// </summary>
public static class DockerCommand
{
    public static Command Create()
    {
        var command = new Command("docker", "Generate Docker configuration for model deployment");

        var modelNameOption = new Option<string?>("--name", "-n")
        {
            Description = "Model name to containerize (defaults to 'default')"
        };

        var portOption = new Option<int>("--port", "-p")
        {
            Description = "Exposed port for API",
            DefaultValueFactory = _ => 5000
        };

        command.Options.Add(modelNameOption);
        command.Options.Add(portOption);

        command.SetAction((parseResult) =>
        {
            var modelName = parseResult.GetValue(modelNameOption) ?? "default";
            var port = parseResult.GetValue(portOption);
            return ExecuteAsync(modelName, port);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(string modelName, int port)
    {
        try
        {
            var fileSystem = new FileSystemManager();
            var projectDiscovery = new ProjectDiscovery(fileSystem);

            // Verify we're in an MLoop project
            string projectRoot;
            try
            {
                projectRoot = projectDiscovery.FindRoot();
            }
            catch (InvalidOperationException)
            {
                ErrorConsole.Error(
                    ProjectDiscovery.NotInsideProjectCause,
                    ProjectDiscovery.NotInsideProjectGuidance);
                return 1;
            }

            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule("[blue]Docker Configuration Generator[/]").LeftJustified());
            AnsiConsole.WriteLine();

            // Generate Dockerfile
            var dockerfilePath = Path.Combine(projectRoot, "Dockerfile");
            var dockerfileContent = GenerateDockerfile(modelName, port);
            await File.WriteAllTextAsync(dockerfilePath, dockerfileContent);

            ValueLine.Write("[green]✓[/] Generated: ", dockerfilePath);

            // Generate .dockerignore
            var dockerignorePath = Path.Combine(projectRoot, ".dockerignore");
            var dockerignoreContent = GenerateDockerignore();
            await File.WriteAllTextAsync(dockerignorePath, dockerignoreContent);

            ValueLine.Write("[green]✓[/] Generated: ", dockerignorePath);

            // Generate docker-compose.yml
            var composePath = Path.Combine(projectRoot, "docker-compose.yml");
            var composeContent = GenerateDockerCompose(modelName, port);
            await File.WriteAllTextAsync(composePath, composeContent);

            ValueLine.Write("[green]✓[/] Generated: ", composePath);

            // Display usage instructions
            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule("[yellow]Usage Instructions[/]").LeftJustified());
            AnsiConsole.WriteLine();

            AnsiConsole.MarkupLine("[bold]Build Docker image:[/]");
            AnsiConsole.MarkupLine($"  docker build -t mloop-{modelName}:latest .");
            AnsiConsole.WriteLine();

            AnsiConsole.MarkupLine("[bold]Run container:[/]");
            AnsiConsole.MarkupLine($"  docker run -p {port}:{port} mloop-{modelName}:latest");
            AnsiConsole.WriteLine();

            AnsiConsole.MarkupLine("[bold]Or use docker-compose (serves in Production, so it requires a signing key):[/]");
            AnsiConsole.MarkupLine("  JWT_SIGNING_KEY=<random secret, 32+ characters> docker compose up");
            AnsiConsole.WriteLine();

            AnsiConsole.MarkupLine("[bold]Test API:[/]");
            AnsiConsole.MarkupLine($"  curl http://localhost:{port}/health");
            AnsiConsole.WriteLine();

            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "docker");
            return 1;
        }
    }

    internal static string GenerateDockerfile(string modelName, int port)
    {
        // The tool is installed in the SDK stage and its output directory copied into the runtime
        // stage. The runtime image already ships Microsoft.NETCore.App and Microsoft.AspNetCore.App,
        // so nothing here installs a .NET distribution — an earlier revision added the Microsoft
        // *Debian 12* package feed to what is an *Ubuntu* image and installed the full SDK on top of
        // the runtime, which could not resolve its own dependencies and failed the build outright.
        // Only wget is installed, because HEALTHCHECK below needs an HTTP client and the runtime
        // image ships neither wget nor curl.
        return $@"# MLoop Model Serving Dockerfile
# Auto-generated by 'mloop docker'

FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build

# The CLI is a dotnet tool, so the build stage exists only to acquire it.
RUN dotnet tool install --global mloop

FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS final
WORKDIR /app
EXPOSE {port}

# Only for HEALTHCHECK — the runtime image has no HTTP client of its own.
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

# Bring the tool over. The runtime it needs is already part of this image.
COPY --from=build /root/.dotnet/tools /root/.dotnet/tools
ENV PATH=""$PATH:/root/.dotnet/tools""

# Model artifacts and configuration
COPY mloop.yaml ./
COPY .mloop/ ./.mloop/
COPY {ExperimentLayout.ModelsDirectory}/ ./{ExperimentLayout.ModelsDirectory}/

ENV ASPNETCORE_URLS=http://+:{port}
ENV MLOOP_PROJECT_ROOT=/app
ENV MLOOP_MODEL_NAME={modelName}

# The environment is left at its default so that `docker run` works with no further setup; the
# API then serves on a development signing key and says so. Deploying this image for real means
# setting both of the following, which is what the generated compose file does:
#   -e ASPNETCORE_ENVIRONMENT=Production -e Jwt__Key=<random secret, 32+ characters>

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 -O /dev/null http://127.0.0.1:{port}/health || exit 1

ENTRYPOINT [""mloop"", ""serve""]
CMD [""--host"", ""0.0.0.0"", ""--port"", ""{port}""]
";
    }

    internal static string GenerateDockerignore()
    {
        return $@"# MLoop Docker Ignore
# Auto-generated by 'mloop docker'

# Ignore temp files
.mloop/temp/
*.tmp

# Ignore logs
*.log

# Ignore OS files
.DS_Store
Thumbs.db

# Ignore IDE files
.vs/
.vscode/
.idea/
*.swp
*.swo

# Ignore git
.git/
.gitignore

# Ignore Docker files
Dockerfile
docker-compose.yml
.dockerignore

# Ignore build artifacts (keep models)
bin/
obj/
!{ExperimentLayout.ModelsDirectory}/
";
    }

    internal static string GenerateDockerCompose(string modelName, int port)
    {
        return $@"# MLoop Docker Compose Configuration
# Auto-generated by 'mloop docker'

services:
  mloop-api:
    build:
      context: .
      dockerfile: Dockerfile
    image: mloop-{modelName}:latest
    container_name: mloop-{modelName}
    ports:
      - ""{port}:{port}""
    environment:
      - ASPNETCORE_ENVIRONMENT=Production
      - MLOOP_MODEL_NAME={modelName}
      # The name deliberately stays out of the MLOOP_ namespace: compose consumes this value and
      # hands it to ASP.NET as Jwt:Key — MLoop itself never reads it, and every MLOOP_ variable the
      # product mentions is one the product honours (GeneratedArtifactContractTests).
      # Production refuses to start on the default signing key, so it is required here rather
      # than defaulted: compose stops with this message when JWT_SIGNING_KEY is unset, instead of
      # starting a container that crash-loops on a fatal it cannot recover from.
      - Jwt__Key=${{JWT_SIGNING_KEY:?set JWT_SIGNING_KEY to a random secret of at least 32 characters}}
    volumes:
      - ./{ExperimentLayout.ModelsDirectory}:/app/{ExperimentLayout.ModelsDirectory}:ro  # Read-only model access
    restart: unless-stopped
    healthcheck:
      test: [""CMD"", ""wget"", ""--quiet"", ""--tries=1"", ""-O"", ""/dev/null"", ""http://127.0.0.1:{port}/health""]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 5s

  # Optional: nginx reverse proxy
  # nginx:
  #   image: nginx:alpine
  #   container_name: mloop-nginx
  #   ports:
  #     - ""80:80""
  #   volumes:
  #     - ./nginx.conf:/etc/nginx/nginx.conf:ro
  #   depends_on:
  #     - mloop-api
  #   restart: unless-stopped
";
    }
}
