using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// Shared context for CLI commands providing common dependencies and initialization.
/// </summary>
public sealed class CommandContext
{
    public IFileSystemManager FileSystem { get; }
    public IProjectDiscovery ProjectDiscovery { get; }
    public string ProjectRoot { get; }
    public IExperimentStore ExperimentStore { get; }
    public IModelRegistry ModelRegistry { get; }
    public ConfigLoader ConfigLoader { get; }

    private CommandContext(
        IFileSystemManager fileSystem,
        IProjectDiscovery projectDiscovery,
        string projectRoot,
        IExperimentStore experimentStore,
        IModelRegistry modelRegistry,
        ConfigLoader configLoader)
    {
        FileSystem = fileSystem;
        ProjectDiscovery = projectDiscovery;
        ProjectRoot = projectRoot;
        ExperimentStore = experimentStore;
        ModelRegistry = modelRegistry;
        ConfigLoader = configLoader;
    }

    /// <summary>
    /// Creates a CommandContext if inside a MLoop project, otherwise displays error and returns null.
    /// </summary>
    /// <returns>CommandContext if successful, null if not in a project</returns>
    public static CommandContext? TryCreate()
    {
        var fileSystem = new FileSystemManager();
        var projectDiscovery = new ProjectDiscovery(fileSystem);

        string projectRoot;
        try
        {
            projectRoot = projectDiscovery.FindRoot();
        }
        catch (InvalidOperationException)
        {
            AnsiConsole.MarkupLine("[red]Error:[/] Not inside a MLoop project.");
            AnsiConsole.MarkupLine("Run [blue]mloop init[/] to create a new project.");
            return null;
        }

        var experimentStore = new ExperimentStore(fileSystem, projectDiscovery);
        var modelRegistry = new ModelRegistry(fileSystem, projectDiscovery, experimentStore);
        var configLoader = new ConfigLoader(fileSystem, projectDiscovery);

        return new CommandContext(
            fileSystem,
            projectDiscovery,
            projectRoot,
            experimentStore,
            modelRegistry,
            configLoader);
    }

    /// <summary>
    /// Resolves a model name, returning the default model name if null/empty.
    /// </summary>
    public static string ResolveModelName(string? modelName)
    {
        return string.IsNullOrWhiteSpace(modelName)
            ? ConfigDefaults.DefaultModelName
            : modelName.Trim().ToLowerInvariant();
    }

    /// <summary>
    /// Resolves a model name, returning null if not specified (for listing all models).
    /// </summary>
    public static string? ResolveOptionalModelName(string? modelName)
    {
        return string.IsNullOrWhiteSpace(modelName)
            ? null
            : modelName.Trim().ToLowerInvariant();
    }
}
