using System.IO.Compression;
using System.Net;
using MLoop.Core.Runtime;

namespace MLoop.Core.Tests.Runtime;

/// <summary>
/// Installing a native runtime — resolve, download, extract, verify, write the manifest — over a stand-in HTTP handler that serves a
/// package built in memory. The success path, every phase it reports, and each way it can end without an install (a refused
/// download, a package without the libraries, a cancelled call) run on any machine, with nothing downloaded.
/// </summary>
public sealed class RuntimeInstallTests : IDisposable
{
    private static readonly string Rid = PlatformDetector.GetRuntimeIdentifier();
    private const string Library = "libfake.so";

    private readonly string _cache = Path.Combine(Path.GetTempPath(), "mloop_runtime_install_" + Guid.NewGuid().ToString("N"));
    private readonly List<string> _requests = [];

    // The install lock is a file named by the runtime id in the shared temp directory: a fresh id per test keeps runs apart.
    private readonly string _id = "fake" + Guid.NewGuid().ToString("N")[..8];

    public void Dispose()
    {
        try { Directory.Delete(_cache, recursive: true); } catch (IOException) { }
        foreach (var leftover in Directory.EnumerateFiles(Path.GetTempPath(), $"mloop-{_id}-*"))
            File.Delete(leftover);
    }

    private RuntimeDefinition Runtime(string version = "1.2.3") => new()
    {
        Id = _id,
        DisplayName = "Fake runtime",
        Description = "",
        Version = version,
        ApproximateSizeMB = 1,
        NuGetPackages = new() { [Rid] = "Fake.Package" },
        NativeLibraries = new() { [Rid] = [Library] },
        NativeSearchPaths = new() { [Rid] = $"runtimes/{Rid}/native" },
        RequiredByTasks = [],
    };

    private static byte[] Package(params (string Path, string Content)[] entries)
    {
        using var stream = new MemoryStream();
        using (var zip = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: true))
        {
            foreach (var (path, content) in entries)
            {
                using var writer = new StreamWriter(zip.CreateEntry(path).Open());
                writer.Write(content);
            }
        }
        return stream.ToArray();
    }

    private sealed class Stub(Func<HttpRequestMessage, HttpResponseMessage> answer) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => Task.FromResult(answer(request));
    }

    private RuntimeManager Serving(byte[]? package)
        => new(_cache, new Stub(request =>
        {
            _requests.Add(request.RequestUri!.ToString());
            return package is null
                ? new HttpResponseMessage(HttpStatusCode.NotFound)
                : new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(package) };
        }));

    private sealed class Collect : IProgress<RuntimeDownloadProgress>
    {
        public List<RuntimeDownloadProgress> Reports { get; } = [];
        public void Report(RuntimeDownloadProgress value) => Reports.Add(value);
    }

    private string RuntimeDir => Path.Combine(_cache, _id, Rid);

    [Fact]
    public async Task AnInstall_ResolvesTheNuGetPackage_ExtractsItsLibraries_WritesTheManifest_AndReportsEveryPhaseInOrder()
    {
        var manager = Serving(Package(($"runtimes/{Rid}/native/{Library}", "native"), ("lib/net8.0/Managed.dll", "managed")));
        var progress = new Collect();
        var runtime = Runtime();

        await manager.InstallAsync(runtime, progress);

        Assert.Equal("https://api.nuget.org/v3-flatcontainer/fake.package/1.2.3/fake.package.1.2.3.nupkg", _requests.Single());
        Assert.Equal("native", await File.ReadAllTextAsync(Path.Combine(RuntimeDir, Library)));
        Assert.False(File.Exists(Path.Combine(RuntimeDir, "Managed.dll")), "only the native directory of the package is unpacked");
        Assert.Equal("1.2.3", manager.InstalledVersion(runtime));
        Assert.True(manager.IsInstalled(runtime));
        Assert.True(manager.GetStatus(runtime).Installed);
        Assert.Equal(
            [DownloadPhase.Resolving, DownloadPhase.Downloading, DownloadPhase.Extracting, DownloadPhase.Verifying, DownloadPhase.Complete],
            progress.Reports.Select(r => r.Phase).Distinct().ToArray());
        Assert.Empty(Directory.EnumerateFiles(Path.GetTempPath(), $"mloop-{_id}-*"));   // the downloaded package is not kept
    }

    [Fact]
    public async Task AnInstalledRuntime_IsNotDownloadedAgain_AndSaysSo()
    {
        var manager = Serving(Package(($"runtimes/{Rid}/native/{Library}", "native")));
        await manager.InstallAsync(Runtime());
        _requests.Clear();
        var progress = new Collect();

        await manager.InstallAsync(Runtime(), progress);

        Assert.Empty(_requests);
        Assert.Contains("already installed", progress.Reports.Last().Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ARefusedDownload_Throws_AndInstallsNothing()
    {
        var manager = Serving(package: null);   // the package answers 404

        await Assert.ThrowsAsync<HttpRequestException>(() => manager.InstallAsync(Runtime()));

        Assert.False(manager.IsInstalled(Runtime()));
        Assert.False(File.Exists(Path.Combine(RuntimeDir, "manifest.txt")));
        Assert.Empty(Directory.EnumerateFiles(Path.GetTempPath(), $"mloop-{_id}-*"));
    }

    [Fact]
    public async Task APackageWithoutTheLibrariesForThisPlatform_IsRefusedByName_AndNothingOfItIsKept()
    {
        var manager = Serving(Package(("runtimes/another-platform/native/libother.so", "x"), ("readme.txt", "x")));

        var failure = await Assert.ThrowsAsync<InvalidOperationException>(() => manager.InstallAsync(Runtime()));

        Assert.Contains("does not carry the native libraries", failure.Message, StringComparison.Ordinal);
        Assert.Contains(Rid, failure.Message, StringComparison.Ordinal);
        Assert.Contains(Library, failure.Message, StringComparison.Ordinal);
        Assert.False(Directory.Exists(RuntimeDir));
    }

    [Fact]
    public async Task APackageThatKeepsItsLibrariesAtTheRoot_IsReadByTheFallback()
    {
        var manager = Serving(Package((Library, "native at the root"), ("notes.txt", "not a library")));

        await manager.InstallAsync(Runtime());

        Assert.Equal("native at the root", await File.ReadAllTextAsync(Path.Combine(RuntimeDir, Library)));
        Assert.False(File.Exists(Path.Combine(RuntimeDir, "notes.txt")));
        Assert.True(manager.IsInstalled(Runtime()));
    }

    [Fact]
    public async Task LibrariesOfAnotherVersion_AreReplacedWhole_NoFileOfTheOldVersionIsLeftBesideTheNew()
    {
        Directory.CreateDirectory(RuntimeDir);
        await File.WriteAllTextAsync(Path.Combine(RuntimeDir, Library), "old native");
        await File.WriteAllTextAsync(Path.Combine(RuntimeDir, "libstale.so"), "only the old version had this");
        await File.WriteAllTextAsync(Path.Combine(RuntimeDir, "manifest.txt"), $"{_id}\n0.9.0\n{Rid}\n2020-01-01");
        var manager = Serving(Package(($"runtimes/{Rid}/native/{Library}", "new native")));
        Assert.False(manager.IsInstalled(Runtime()), "a library of another version counts as not installed");

        await manager.InstallAsync(Runtime());

        Assert.Equal("new native", await File.ReadAllTextAsync(Path.Combine(RuntimeDir, Library)));
        Assert.False(File.Exists(Path.Combine(RuntimeDir, "libstale.so")));
        Assert.Equal("1.2.3", manager.InstalledVersion(Runtime()));
    }

    [Fact]
    public async Task ACancelledInstall_IsCancelled_AndDownloadsNothing()
    {
        var manager = Serving(Package(($"runtimes/{Rid}/native/{Library}", "native")));
        using var cancelled = new CancellationTokenSource();
        await cancelled.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => manager.InstallAsync(Runtime(), cancellationToken: cancelled.Token));

        Assert.Empty(_requests);
        Assert.False(manager.IsInstalled(Runtime()));
    }

    [Fact]
    public async Task ARemovedRuntime_IsNoLongerInstalled()
    {
        var manager = Serving(Package(($"runtimes/{Rid}/native/{Library}", "native")));
        await manager.InstallAsync(Runtime());

        manager.Remove(Runtime());

        Assert.False(manager.IsInstalled(Runtime()));
        Assert.False(Directory.Exists(RuntimeDir));
    }
}
