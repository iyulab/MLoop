using System.CommandLine;
using System.Diagnostics;
using MLoop.CLI.Infrastructure;
using MLoop.CLI.Infrastructure.Diagnostics;
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
        var portOption = new Option<int>("--port", "-p")
        {
            Description = "Port to run the server on",
            DefaultValueFactory = _ => 5000
        };

        var hostOption = new Option<string>("--host", "-h")
        {
            Description = "Host address to bind to",
            DefaultValueFactory = _ => "localhost"
        };

        var detachOption = new Option<bool>("--detach", "-d")
        {
            Description = "Run server in background",
            DefaultValueFactory = _ => false
        };

        this.Options.Add(portOption);
        this.Options.Add(hostOption);
        this.Options.Add(detachOption);

        this.SetAction((parseResult) =>
        {
            var port = parseResult.GetValue(portOption);
            var host = parseResult.GetValue(hostOption)!;
            var detach = parseResult.GetValue(detachOption);

            // Initialize services
            var fileSystem = new FileSystemManager();
            var projectDiscovery = new ProjectDiscovery(fileSystem);

            return ExecuteAsync(port, host, detach, projectDiscovery);
        });
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
                AnsiConsole.MarkupLine("[red]❌ MLoop.API assembly not found.[/]");
                AnsiConsole.WriteLine();
                AnsiConsole.MarkupLine("[yellow]Options:[/]");
                AnsiConsole.MarkupLine("[grey]  1. Set MLOOP_API_PATH environment variable:[/]");
                AnsiConsole.MarkupLine("[grey]     set MLOOP_API_PATH=D:\\path\\to\\MLoop.API.dll[/]");
                AnsiConsole.WriteLine();
                AnsiConsole.MarkupLine("[grey]  2. Build from source:[/]");
                AnsiConsole.MarkupLine("[grey]     dotnet build src/MLoop.sln[/]");
                AnsiConsole.MarkupLine("[grey]     dotnet run --project tools/MLoop.API[/]");
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
            ErrorSuggestions.DisplayError(ex, "serve");
            return;
        }
    }

    private static string? FindApiAssembly()
    {
        // 1. Check MLOOP_API_PATH environment variable first (highest priority)
        var envPath = Environment.GetEnvironmentVariable("MLOOP_API_PATH");
        if (!string.IsNullOrEmpty(envPath) && File.Exists(envPath))
        {
            return Path.GetFullPath(envPath);
        }

        // 2. Look for MLoop.API.dll in common locations
        var searchPaths = new List<string>
        {
            // Same directory as CLI (for bundled deployments)
            Path.Combine(AppContext.BaseDirectory, "MLoop.API.dll"),
            // Relative paths from CLI location
            Path.Combine(AppContext.BaseDirectory, "..", "MLoop.API", "MLoop.API.dll"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "MLoop.API", "MLoop.API.dll"),
        };

        // 3. Search development build output with dynamic TFM detection
        var devRoots = new[]
        {
            // From CLI bin directory: navigate up to solution root
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."),
            // From current working directory
            Directory.GetCurrentDirectory(),
        };

        foreach (var root in devRoots)
        {
            var apiProjectDir = Path.Combine(root, "tools", "MLoop.API", "bin");
            if (Directory.Exists(apiProjectDir))
            {
                // Search across all configurations and target frameworks
                foreach (var config in new[] { "Debug", "Release" })
                {
                    var configDir = Path.Combine(apiProjectDir, config);
                    if (!Directory.Exists(configDir)) continue;

                    // Find net* directories dynamically (net9.0, net10.0, etc.)
                    try
                    {
                        foreach (var tfmDir in Directory.GetDirectories(configDir, "net*"))
                        {
                            searchPaths.Add(Path.Combine(tfmDir, "MLoop.API.dll"));
                        }
                    }
                    catch (IOException)
                    {
                        // Directory access error, skip
                    }
                }
            }
        }

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
