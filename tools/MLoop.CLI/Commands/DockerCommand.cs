using System.CommandLine;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.Update;
using MLoop.Core.Embedding;
using MLoop.Core.ImageAnomaly;
using MLoop.Core.Models;
using MLoop.Core.Storage;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// Docker command to generate containerization files
/// </summary>
public static class DockerCommand
{
    /// <summary>
    /// The host directory the compose file mounts for encoders: wherever <see cref="EncoderStore.RootVariable"/> points when
    /// compose runs, else the home-relative default — the same rule <see cref="EncoderStore.DefaultRoot"/> applies on the
    /// host that installed them. The file is committed and run on other machines, so it names the variable, not a path.
    /// </summary>
    private static readonly string ComposeEncodersHost = "${" + EncoderStore.RootVariable + ":-" + EncoderStore.HomeRelativeRoot + "}";

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

            // A model that reads a text column through an encoder, or scores pictures against a bank an encoder made, cannot
            // run without that encoder and ONNX Runtime, so the image is generated with both switched on rather than left
            // to fail at the first prediction.
            var textEncoders = EncodersUnder(projectRoot);

            // Generate Dockerfile
            var dockerfilePath = Path.Combine(projectRoot, "Dockerfile");
            var dockerfileContent = GenerateDockerfile(modelName, port, textEncoders);
            await File.WriteAllTextAsync(dockerfilePath, dockerfileContent);

            ValueLine.Write("[green]✓[/] Generated: ", dockerfilePath);

            // Generate .dockerignore
            var dockerignorePath = Path.Combine(projectRoot, ".dockerignore");
            var dockerignoreContent = GenerateDockerignore();
            await File.WriteAllTextAsync(dockerignorePath, dockerignoreContent);

            ValueLine.Write("[green]✓[/] Generated: ", dockerignorePath);

            // Generate docker-compose.yml
            var composePath = Path.Combine(projectRoot, "docker-compose.yml");
            var composeContent = GenerateDockerCompose(modelName, port, textEncoders);
            await File.WriteAllTextAsync(composePath, composeContent);

            ValueLine.Write("[green]✓[/] Generated: ", composePath);

            // Display usage instructions
            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule("[yellow]Usage Instructions[/]").LeftJustified());
            AnsiConsole.WriteLine();

            if (textEncoders.Count > 0)
            {
                AnsiConsole.MarkupLine("[bold]Encoders:[/] the models run through encoders the image does not carry —");
                AnsiConsole.MarkupLine($"  have them on this host (the container mounts {Markup.Escape(EncoderStore.DefaultRoot)}):");
                foreach (var encoder in textEncoders)
                    ValueLine.Write(Markup.Escape($"  {encoder.Id}: "), EncoderStore.HowToInstall(encoder.Id, encoder.Kind));
                AnsiConsole.WriteLine();
            }

            AnsiConsole.MarkupLine("[bold]Build Docker image:[/]");
            AnsiConsole.MarkupLine($"  docker build -t mloop-{modelName}:latest .");
            AnsiConsole.WriteLine();

            AnsiConsole.MarkupLine("[bold]Run container:[/]");
            AnsiConsole.MarkupLine(textEncoders.Count > 0
                ? $"  docker run -p {port}:{port} -v \"{Markup.Escape(EncoderStore.DefaultRoot)}:/encoders:ro\" mloop-{modelName}:latest"
                : $"  docker run -p {port}:{port} mloop-{modelName}:latest");
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

    /// <summary>
    /// The encoders the models under <paramref name="projectRoot"/> need to run — the text encoders a model was trained with
    /// (<see cref="TextEncodings"/> beside each experiment and production model) and the encoder an image-anomaly model's bank
    /// was made with (<see cref="ImageAnomalyModel"/>) — one per id. The image copies every model, so any of them can be served.
    /// </summary>
    internal static IReadOnlyList<EncoderIdentity> EncodersUnder(string projectRoot)
    {
        var models = Path.Combine(projectRoot, ExperimentLayout.ModelsDirectory);
        if (!Directory.Exists(models))
            return [];
        string ModelBeside(string path) => Path.Combine(Path.GetDirectoryName(path)!, ExperimentLayout.ModelFileName);

        var text = Directory.EnumerateFiles(models, ExperimentLayout.TextEncodersFileName, SearchOption.AllDirectories)
            .Select(path => TextEncodings.ReadBeside(ModelBeside(path)))
            .SelectMany(encodings => encodings?.Columns ?? [])
            .Select(column => column.Encoder);
        var pictures = Directory.EnumerateFiles(models, ExperimentLayout.ImageAnomalyFileName, SearchOption.AllDirectories)
            .Select(path => ImageAnomalyModel.ReadBeside(ModelBeside(path)))
            .OfType<ImageAnomalyModel>()
            .Select(model => model.Encoder);
        return text.Concat(pictures)
            .DistinctBy(encoder => encoder.Id, StringComparer.OrdinalIgnoreCase)
            .OrderBy(encoder => encoder.Id, StringComparer.Ordinal)
            .ToList();
    }

    internal static string GenerateDockerfile(string modelName, int port, IReadOnlyList<EncoderIdentity>? textEncoders = null)
    {
        var encoded = textEncoders is { Count: > 0 };
        // The tool is installed in the SDK stage and its output directory copied into the runtime
        // stage. The runtime image already ships Microsoft.NETCore.App and Microsoft.AspNetCore.App,
        // so nothing here installs a .NET distribution — an earlier revision added the Microsoft
        // *Debian 12* package feed to what is an *Ubuntu* image and installed the full SDK on top of
        // the runtime, which could not resolve its own dependencies and failed the build outright.
        // Two packages are installed: wget, because HEALTHCHECK below needs an HTTP client and the
        // runtime image ships neither wget nor curl, and libomp5, because ML.NET's time-series
        // algorithms load a native library that expects a system OpenMP runtime on Linux.
        return $@"# MLoop Model Serving Dockerfile
# Auto-generated by 'mloop docker'

FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build

# The CLI is a dotnet tool, so the build stage exists only to acquire it. The version is pinned to
# the one that generated this file, so rebuilding the image a year from now still produces the
# image you tested — an unpinned install silently follows whatever is newest on nuget.org, which
# means the model in this directory and the CLI serving it can drift apart without anything saying
# so. Override for a different one:  docker build --build-arg CLI_VERSION=x.y.z .
#
# The name deliberately stays out of the MLOOP_ namespace: this is a build argument docker consumes
# while assembling the image, not a variable the running product reads — and every MLOOP_ variable a
# generated file mentions is one the product honours (GeneratedArtifactContractTests).
ARG CLI_VERSION={UpdateChecker.GetCurrentVersion()}
RUN dotnet tool install --global mloop --version ""$CLI_VERSION""

FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS final
WORKDIR /app
EXPOSE {port}

# wget is for HEALTHCHECK below — the runtime image ships no HTTP client of its own.
# libomp5 is for ML.NET's time-series algorithms (forecasting, time-series anomaly detection).
# Their FFT comes from the MKL native library, which on Linux is published without the OpenMP
# runtime it links against: ML.NET expects the system to provide one. LLVM's is ABI-compatible,
# so exposing it under the name the loader asks for is what makes `POST /train` work for those
# tasks inside this image instead of failing to load a shared library.
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget libomp5 \
    && libomp=""$(ldconfig -p | awk '/libomp\.so\.5/ {{print $NF; exit}}')"" \
    && test -n ""$libomp"" \
    && ln -sf ""$libomp"" ""$(dirname ""$libomp"")/libiomp5.so"" \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

# Bring the tool over. The runtime it needs is already part of this image.
COPY --from=build /root/.dotnet/tools /root/.dotnet/tools
ENV PATH=""$PATH:/root/.dotnet/tools""

# Model artifacts and configuration
COPY mloop.yaml ./
COPY .mloop/ ./.mloop/
COPY {ExperimentLayout.ModelsDirectory}/ ./{ExperimentLayout.ModelsDirectory}/

{(encoded ? $@"# The models run through encoders ({string.Join(", ", textEncoders!.Select(e => e.Id))}), which run on ONNX Runtime
# (~{MLoop.Core.Runtime.RuntimeRegistry.OnnxRuntime.ApproximateSizeMB} MB). The encoders themselves are not in the image: they are mounted, as docker-compose.yml shows.
RUN mloop runtime install {MLoop.Core.Runtime.RuntimeRegistry.OnnxRuntime.Id} --quiet
ENV {EncoderStore.RootVariable}=/encoders" : $@"# POST /embed (pretrained image and text encoders) runs on ONNX Runtime, whose native library this image carries only when
# asked — uncomment to install it (~{MLoop.Core.Runtime.RuntimeRegistry.OnnxRuntime.ApproximateSizeMB} MB), and mount the encoders as docker-compose.yml shows:
# RUN mloop runtime install {MLoop.Core.Runtime.RuntimeRegistry.OnnxRuntime.Id} --quiet")}

ENV ASPNETCORE_URLS=http://+:{port}
ENV MLOOP_PROJECT_ROOT=/app
ENV {MLoop.Core.Storage.ModelName.EnvironmentVariable}={modelName}

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

    internal static string GenerateDockerCompose(string modelName, int port, IReadOnlyList<EncoderIdentity>? textEncoders = null)
    {
        var encoded = textEncoders is { Count: > 0 };
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
      - {MLoop.Core.Storage.ModelName.EnvironmentVariable}={modelName}
      # The name deliberately stays out of the MLOOP_ namespace: compose consumes this value and
      # hands it to ASP.NET as Jwt:Key — MLoop itself never reads it, and every MLOOP_ variable the
      # product mentions is one the product honours (GeneratedArtifactContractTests).
      # Production refuses to start on the default signing key, so it is required here rather
      # than defaulted: compose stops with this message when JWT_SIGNING_KEY is unset, instead of
      # starting a container that crash-loops on a fatal it cannot recover from.
      - Jwt__Key=${{JWT_SIGNING_KEY:?set JWT_SIGNING_KEY to a random secret of at least 32 characters}}
{(encoded ? $@"      # The models run through encoders ({string.Join(", ", textEncoders!.Select(e => e.Id))}): the ones `mloop encoder install`
      # put on the host, mounted (the image carries none). A model refuses to load when its encoder is missing or another.
      - {EncoderStore.RootVariable}=/encoders" : $@"      # POST /embed: the encoders `mloop encoder install` (or `add`) put on the host, mounted (the image carries none), with the
      # Dockerfile's ONNX Runtime line uncommented:
      # - {EncoderStore.RootVariable}=/encoders")}
    volumes:
      - ./{ExperimentLayout.ModelsDirectory}:/app/{ExperimentLayout.ModelsDirectory}:ro  # Read-only model access
      {(encoded ? "" : "# ")}- {ComposeEncodersHost}:/encoders:ro
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
