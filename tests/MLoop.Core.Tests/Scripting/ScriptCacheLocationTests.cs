using MLoop.Core.Scripting;

namespace MLoop.Core.Tests.Scripting;

/// <summary>
/// A project's compiled scripts are cached inside that project, not beside whatever directory the
/// command was invoked from.
/// </summary>
/// <remarks>
/// The CLI never changes its working directory — it resolves the project root and passes it around
/// — so a cache path left relative resolves against the user's shell instead. Running
/// <c>mloop train</c> from a subdirectory then wrote <c>.mloop/.cache/scripts</c> into that
/// subdirectory, left it there, and recompiled the same scripts on the next run from anywhere else.
/// Every caller had the project root in hand and dropped it.
/// </remarks>
public class ScriptCacheLocationTests : IDisposable
{
    private readonly string _projectRoot =
        Path.Combine(Path.GetTempPath(), "mloop_cache_location_" + Guid.NewGuid().ToString("N"));

    public ScriptCacheLocationTests() => Directory.CreateDirectory(_projectRoot);

    public void Dispose()
    {
        GC.SuppressFinalize(this);
        if (Directory.Exists(_projectRoot))
            Directory.Delete(_projectRoot, recursive: true);
    }

    [Fact]
    public void TheCacheDirectoryIsInsideTheProject()
    {
        var cache = ScriptLoader.CacheDirectoryFor(_projectRoot);

        Assert.True(Path.IsPathFullyQualified(cache), $"expected an absolute path, got '{cache}'");
        Assert.StartsWith(_projectRoot, cache, StringComparison.Ordinal);
    }

    [Fact]
    public async Task DiscoveryCachesIntoTheProjectItWasGiven()
    {
        var hooks = Path.Combine(_projectRoot, ".mloop", "scripts", "hooks");
        Directory.CreateDirectory(hooks);
        await File.WriteAllTextAsync(Path.Combine(hooks, "Noop.cs"), """
            using System.Threading.Tasks;
            using MLoop.Extensibility.Hooks;

            public class NoopHook : IMLoopHook
            {
                public string Name => "noop";
                public Task<HookResult> ExecuteAsync(HookContext context) =>
                    Task.FromResult(HookResult.Continue());
            }
            """);

        // No working-directory change: the point is that the project root alone decides where the
        // cache lands, and the runner's own directory is not the project.
        var discovered = await new ScriptDiscovery(_projectRoot, log: _ => { }).DiscoverHooksAsync();

        Assert.Single(discovered);
        Assert.True(
            Directory.Exists(ScriptLoader.CacheDirectoryFor(_projectRoot)),
            "compiling a project's script did not cache inside that project — the cache went "
            + "wherever the process happened to be running from");
    }
}
