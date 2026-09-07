using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.Infrastructure.FileSystem;

/// <summary>
/// The two environment variables MLoop's own generated Dockerfile sets are read.
/// </summary>
/// <remarks>
/// <c>mloop docker</c> writes <c>ENV MLOOP_PROJECT_ROOT=/app</c> and
/// <c>ENV MLOOP_MODEL_NAME=&lt;model&gt;</c>, and <c>mloop serve</c> sets the first as well — and
/// nothing read either. The images worked only because their WORKDIR happened to be the project and
/// the model happened to be named <c>default</c>; a deployment that relied on what the file says
/// would have been served something else without being told. These tests are the reason that cannot
/// go back to being decoration.
///
/// The variables are process-wide, which is why this assembly disables test parallelization; each
/// test restores what it found.
/// </remarks>
public class EnvironmentConfigurationTests
{
    [Fact]
    public void ProjectRootVariableTakesPrecedenceOverTheWorkingDirectory()
    {
        var project = CreateProject();
        using var _ = new EnvironmentVariable(ProjectDiscovery.ProjectRootVariable, project);
        try
        {
            var root = new ProjectDiscovery(new FileSystemManager()).FindRoot();

            Assert.Equal(project, root.TrimEnd(Path.DirectorySeparatorChar));
        }
        finally
        {
            Directory.Delete(project, recursive: true);
        }
    }

    [Fact]
    public void ProjectRootVariableNamingSomethingElseFailsByName()
    {
        var notAProject = Path.Combine(Path.GetTempPath(), "mloop_not_a_project_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(notAProject);
        using var _ = new EnvironmentVariable(ProjectDiscovery.ProjectRootVariable, notAProject);
        try
        {
            var error = Assert.Throws<InvalidOperationException>(
                () => new ProjectDiscovery(new FileSystemManager()).FindRoot());

            // Naming the variable is the whole point: a wrong value is an operator's typo, and the
            // message has to say which knob to look at rather than describing the directory it landed in.
            Assert.Contains(ProjectDiscovery.ProjectRootVariable, error.Message, StringComparison.Ordinal);
            Assert.Contains(notAProject, error.Message, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(notAProject, recursive: true);
        }
    }

    [Fact]
    public void ModelNameVariableSuppliesTheDefaultModel()
    {
        using var _ = new EnvironmentVariable(ModelNameResolver.ModelNameVariable, "Fraud-Detector");

        // Resolve is pure with respect to the resolver's project state, so the name alone is enough.
        Assert.Equal("fraud-detector", ResolveWithNoNameGiven());
    }

    [Fact]
    public void AnExplicitNameStillWinsOverTheVariable()
    {
        using var _ = new EnvironmentVariable(ModelNameResolver.ModelNameVariable, "Fraud-Detector");

        Assert.Equal("churn", ResolveWithName("Churn"));
    }

    [Fact]
    public void WithoutTheVariableTheDefaultIsUnchanged()
    {
        using var _ = new EnvironmentVariable(ModelNameResolver.ModelNameVariable, null);

        Assert.Equal(ConfigDefaults.DefaultModelName, ResolveWithNoNameGiven());
    }

    private static string ResolveWithNoNameGiven() => ResolveWithName(null);

    private static string ResolveWithName(string? name)
    {
        var project = CreateProject();
        try
        {
            using var _ = new EnvironmentVariable(ProjectDiscovery.ProjectRootVariable, project);
            var fileSystem = new FileSystemManager();
            var discovery = new ProjectDiscovery(fileSystem);
            var resolver = new ModelNameResolver(fileSystem, discovery, new ConfigLoader(fileSystem, discovery));
            return resolver.Resolve(name);
        }
        finally
        {
            Directory.Delete(project, recursive: true);
        }
    }

    private static string CreateProject()
    {
        var root = Path.Combine(Path.GetTempPath(), "mloop_env_config_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(root, ".mloop"));
        return root;
    }

    /// <summary>Sets a process-wide variable for the duration of a test and puts back what was there.</summary>
    private sealed class EnvironmentVariable : IDisposable
    {
        private readonly string _name;
        private readonly string? _previous;

        public EnvironmentVariable(string name, string? value)
        {
            _name = name;
            _previous = Environment.GetEnvironmentVariable(name);
            Environment.SetEnvironmentVariable(name, value);
        }

        public void Dispose() => Environment.SetEnvironmentVariable(_name, _previous);
    }
}
