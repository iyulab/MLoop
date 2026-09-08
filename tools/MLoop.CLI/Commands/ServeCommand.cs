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
    public ServeCommand() : base("serve",
        "Start REST API server for model serving. All endpoints except /health require a JWT bearer " +
        "token — mint one with `mloop token` (add `--role admin` for write endpoints) and send it as " +
        "the `Authorization: Bearer <token>` header. Set `Jwt:Key` for production.")
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

    /// <summary>
    /// Returns the process exit code: <c>0</c> when the server ran and stopped normally, non-zero
    /// when it could not start or the child API exited with a failure.
    /// </summary>
    /// <remarks>
    /// This returned <c>Task</c> rather than <c>Task&lt;int&gt;</c>, so every failure — a missing
    /// API assembly, an unhandled exception, a child that refused to start — printed a diagnostic
    /// and then exited <c>0</c>. A supervisor or a CI step that starts the server and branches on
    /// the exit code was told the server was running whenever <c>mloop serve</c> returned at all.
    /// <c>ErrorSuggestions.DisplayError</c> already documents the CLI-wide contract it serves
    /// ("every command's top-level catch funnels here before returning a non-zero exit code");
    /// serve was the command that did not hold up its half.
    /// </remarks>
    private static async Task<int> ExecuteAsync(
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
                // stderr, not stdout: this is why the command is about to exit non-zero, and a
                // caller reading the failure reason reads stderr. The lines below said the same
                // thing on the success channel, where a supervisor never looks.
                ErrorConsole.Error(
                    "MLoop.API assembly not found.",
                    "Point MLOOP_API_PATH at the built assembly (MLOOP_API_PATH=<path>/MLoop.API.dll), "
                    + "or build from source: dotnet build MLoop.slnx");
                return 1;
            }

            // Set environment variable for project root
            Environment.SetEnvironmentVariable("MLOOP_PROJECT_ROOT", projectRoot);

            // `mloop serve` is a local serving convenience (binds localhost by default), so default the
            // child API to the Development environment unless the user explicitly set one. Otherwise the
            // API starts in Production, rejects the default JWT key, and exits immediately — making
            // `mloop serve` unusable out of the box. Production deployments set ASPNETCORE_ENVIRONMENT
            // (and a real Jwt:Key) themselves; this only fills the unset case.
            if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT")))
            {
                Environment.SetEnvironmentVariable("ASPNETCORE_ENVIRONMENT", "Development");
            }

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
                        AnsiConsole.MarkupLine($"[grey]{Markup.Escape(e.Data)}[/]");
                    }
                };

                // The child's stderr stays stderr. Folding it into the parent's stdout made the
                // API's own failure lines indistinguishable from its startup log for anything
                // reading the streams rather than watching them.
                process.ErrorDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        ErrorConsole.Out.MarkupLine($"[red]{Markup.Escape(e.Data)}[/]");
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

                // The child is the server; if it failed, serve failed. Reporting 0 here would tell
                // a supervisor the run was clean whenever the API crashed after starting.
                if (process.ExitCode != 0)
                {
                    ErrorConsole.Error(
                        $"MLoop.API exited with code {process.ExitCode}.",
                        "The server's own output above carries the cause.");
                    return process.ExitCode;
                }
            }

            return 0;
        }
        catch (OperationCanceledException)
        {
            // A deliberate stop is not a failure.
            AnsiConsole.MarkupLine("\n[yellow]👋 Server stopped by user[/]");
            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "serve");
            return 1;
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
            // Same directory as CLI (for bundled deployments). The packed `mloop`
            // dotnet tool bundles MLoop.API here via MLoop.CLI.csproj's
            // BundleApiIntoTool target, so installed-tool serve works out-of-box.
            Path.Combine(AppContext.BaseDirectory, "MLoop.API.dll"),
            // Relative paths from CLI location
            Path.Combine(AppContext.BaseDirectory, "..", "MLoop.API", "MLoop.API.dll"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "MLoop.API", "MLoop.API.dll"),
        };

        // 2b. Fixed deployment locations are deterministic — first existing wins.
        foreach (var path in searchPaths)
        {
            var fullPath = Path.GetFullPath(path);
            if (File.Exists(fullPath))
            {
                return fullPath;
            }
        }

        // 3. Search development build output with dynamic TFM detection.
        var devRoots = new[]
        {
            // From CLI bin directory: navigate up to solution root
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."),
            // From current working directory
            Directory.GetCurrentDirectory(),
        };

        var devCandidates = new List<string>();
        foreach (var root in devRoots)
        {
            var apiProjectDir = Path.Combine(root, "tools", "MLoop.API", "bin");
            if (!Directory.Exists(apiProjectDir)) continue;

            // Collect every built MLoop.API.dll across all configurations and target frameworks.
            foreach (var config in new[] { "Debug", "Release" })
            {
                var configDir = Path.Combine(apiProjectDir, config);
                if (!Directory.Exists(configDir)) continue;

                // Find net* directories dynamically (net9.0, net10.0, etc.)
                try
                {
                    foreach (var tfmDir in Directory.GetDirectories(configDir, "net*"))
                    {
                        devCandidates.Add(Path.GetFullPath(Path.Combine(tfmDir, "MLoop.API.dll")));
                    }
                }
                catch (IOException)
                {
                    // Directory access error, skip
                }
            }
        }

        // D9: among dev builds, prefer the most recently built — a fresh Release build must win
        // over a stale Debug one. The old "first existing, Debug before Release" order loaded the
        // stale Debug assembly, hiding fixes built into Release, which is why MLOOP_API_PATH exists.
        return NewestExisting(devCandidates, File.Exists, File.GetLastWriteTimeUtc);
    }

    /// <summary>
    /// Returns the existing candidate with the most recent last-write time, or <see langword="null"/>
    /// when none exist. Pure selection policy (filesystem access injected) so the
    /// newest-build-wins rule is unit-testable. <paramref name="exists"/>/<paramref name="lastWriteUtc"/>
    /// default to the real filesystem.
    /// </summary>
    internal static string? NewestExisting(
        IEnumerable<string> candidates,
        Func<string, bool> exists,
        Func<string, DateTime> lastWriteUtc)
    {
        string? best = null;
        var bestTime = DateTime.MinValue;
        foreach (var candidate in candidates)
        {
            if (!exists(candidate)) continue;
            var time = lastWriteUtc(candidate);
            if (best is null || time > bestTime)
            {
                best = candidate;
                bestTime = time;
            }
        }
        return best;
    }
}
