using System.CommandLine;
using System.Diagnostics;
using MLoop.CLI.Infrastructure;
using MLoop.CLI.Infrastructure.FileSystem;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// Serve command to start REST API for model serving
/// </summary>
public class ServeCommand : Command
{
    public ServeCommand() : base("serve", "Start REST API server for model serving")
    {
        var portOption = new Option<int>(
            name: "--port",
            description: "Port to run the server on",
            getDefaultValue: () => 5000);

        var hostOption = new Option<string>(
            name: "--host",
            description: "Host address to bind to",
            getDefaultValue: () => "localhost");

        var detachOption = new Option<bool>(
            name: "--detach",
            description: "Run server in background",
            getDefaultValue: () => false);

        AddOption(portOption);
        AddOption(hostOption);
        AddOption(detachOption);

        this.SetHandler(async (port, host, detach) =>
        {
            // Initialize services
            var fileSystem = new FileSystemManager();
            var projectDiscovery = new ProjectDiscovery(fileSystem);

            await ExecuteAsync(port, host, detach, projectDiscovery);
        }, portOption, hostOption, detachOption);
    }

    private static async Task ExecuteAsync(
        int port,
        string host,
        bool detach,
        IProjectDiscovery projectDiscovery)
    {
        try
        {
            // Verify we're in an MLoop project
            var projectRoot = projectDiscovery.FindRoot();

            AnsiConsole.MarkupLine("[bold blue]🚀 Starting MLoop API Server...[/]");
            AnsiConsole.MarkupLine($"[grey]📂 Project: {projectRoot}[/]");
            AnsiConsole.MarkupLine($"[grey]🌐 Address: http://{host}:{port}[/]");
            AnsiConsole.WriteLine();

            // Find the MLoop.API executable
            var apiAssembly = FindApiAssembly();

            if (apiAssembly == null)
            {
                AnsiConsole.MarkupLine("[red]❌ MLoop.API assembly not found. Build the solution first:[/]");
                AnsiConsole.MarkupLine("[yellow]   dotnet build[/]");
                return;
            }

            // Set environment variable for project root
            Environment.SetEnvironmentVariable("MLOOP_PROJECT_ROOT", projectRoot);

            // Start the API server
            var startInfo = new ProcessStartInfo
            {
                FileName = "dotnet",
                Arguments = $"\"{apiAssembly}\" --urls http://{host}:{port}",
                UseShellExecute = detach,
                CreateNoWindow = !detach,
                WorkingDirectory = projectRoot
            };

            if (!detach)
            {
                startInfo.RedirectStandardOutput = true;
                startInfo.RedirectStandardError = true;
            }

            var process = new Process { StartInfo = startInfo };

            if (!detach)
            {
                // Attach output handlers for interactive mode
                process.OutputDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        AnsiConsole.MarkupLine($"[grey]{e.Data}[/]");
                    }
                };

                process.ErrorDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        AnsiConsole.MarkupLine($"[red]{e.Data}[/]");
                    }
                };
            }

            process.Start();

            if (detach)
            {
                AnsiConsole.MarkupLine($"[green]✅ Server started in background (PID: {process.Id})[/]");
                AnsiConsole.MarkupLine($"[blue]📖 API Documentation: http://{host}:{port}/swagger[/]");
                AnsiConsole.MarkupLine($"[blue]❤️  Health Check: http://{host}:{port}/health[/]");
                AnsiConsole.WriteLine();
                AnsiConsole.MarkupLine("[yellow]To stop the server:[/]");
                AnsiConsole.MarkupLine($"[grey]   kill {process.Id}  # Unix/macOS[/]");
                AnsiConsole.MarkupLine($"[grey]   taskkill /PID {process.Id} /F  # Windows[/]");
            }
            else
            {
                AnsiConsole.MarkupLine("[green]✅ Server started. Press Ctrl+C to stop.[/]");
                AnsiConsole.MarkupLine($"[blue]📖 API Documentation: http://{host}:{port}/swagger[/]");
                AnsiConsole.MarkupLine($"[blue]❤️  Health Check: http://{host}:{port}/health[/]");
                AnsiConsole.MarkupLine($"[blue]📊 Model Info: http://{host}:{port}/info[/]");
                AnsiConsole.MarkupLine($"[blue]🔮 Predictions: POST http://{host}:{port}/predict[/]");
                AnsiConsole.WriteLine();

                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                await process.WaitForExitAsync();
            }
        }
        catch (OperationCanceledException)
        {
            AnsiConsole.MarkupLine("\n[yellow]👋 Server stopped by user[/]");
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]❌ Failed to start server: {ex.Message}[/]");
            throw;
        }
    }

    private static string? FindApiAssembly()
    {
        // Look for MLoop.API.dll in common build output locations
        var searchPaths = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "..", "..", "MLoop.API", "MLoop.API.dll"),
            Path.Combine(AppContext.BaseDirectory, "..", "MLoop.API", "MLoop.API.dll"),
            Path.Combine(AppContext.BaseDirectory, "MLoop.API.dll"),
            Path.Combine(Directory.GetCurrentDirectory(), "src", "MLoop.API", "bin", "Debug", "net9.0", "MLoop.API.dll"),
            Path.Combine(Directory.GetCurrentDirectory(), "src", "MLoop.API", "bin", "Release", "net9.0", "MLoop.API.dll"),
        };

        foreach (var path in searchPaths)
        {
            var fullPath = Path.GetFullPath(path);
            if (File.Exists(fullPath))
            {
                return fullPath;
            }
        }

        return null;
    }
}
