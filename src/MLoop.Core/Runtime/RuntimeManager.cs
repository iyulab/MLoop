using System.IO.Compression;
using System.Runtime.InteropServices;

namespace MLoop.Core.Runtime;

/// <summary>
/// Manages on-demand download and loading of native ML runtimes.
/// Pattern inspired by lm-supply's RuntimeManager.
/// </summary>
public class RuntimeManager
{
    private static readonly string DefaultCacheDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".mloop", "runtimes");

    private readonly string _cacheDir;
    private readonly HttpClient _httpClient;
    private static readonly HashSet<string> _loadedRuntimes = new(StringComparer.OrdinalIgnoreCase);

    public RuntimeManager(string? cacheDir = null)
    {
        _cacheDir = cacheDir ?? DefaultCacheDir;
        _httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(30) };
    }

    /// <summary>
    /// Checks if a runtime is installed (cached) for the current platform
    /// </summary>
    public bool IsInstalled(RuntimeDefinition runtime)
    {
        var rid = PlatformDetector.GetRuntimeIdentifier();
        var runtimeDir = GetRuntimeDir(runtime, rid);

        if (!runtime.NativeLibraries.TryGetValue(rid, out var expectedLibs))
            return false;

        return expectedLibs.Length > 0 && expectedLibs.All(lib =>
            File.Exists(Path.Combine(runtimeDir, lib)));
    }

    /// <summary>
    /// Gets installed runtime info (version, size, path)
    /// </summary>
    public RuntimeStatus GetStatus(RuntimeDefinition runtime)
    {
        var rid = PlatformDetector.GetRuntimeIdentifier();
        var runtimeDir = GetRuntimeDir(runtime, rid);
        var installed = IsInstalled(runtime);

        long sizeBytes = 0;
        if (installed && Directory.Exists(runtimeDir))
        {
            sizeBytes = new DirectoryInfo(runtimeDir)
                .EnumerateFiles("*", SearchOption.AllDirectories)
                .Sum(f => f.Length);
        }

        return new RuntimeStatus
        {
            Runtime = runtime,
            Installed = installed,
            Path = runtimeDir,
            SizeBytes = sizeBytes,
            RuntimeIdentifier = rid
        };
    }

    /// <summary>
    /// Downloads and installs a runtime for the current platform
    /// </summary>
    public async Task InstallAsync(
        RuntimeDefinition runtime,
        IProgress<RuntimeDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var rid = PlatformDetector.GetRuntimeIdentifier();

        if (!runtime.NuGetPackages.TryGetValue(rid, out var packageId))
            throw new PlatformNotSupportedException($"Runtime '{runtime.Id}' is not available for {rid}");

        var runtimeDir = GetRuntimeDir(runtime, rid);

        // Cross-process lock to prevent concurrent downloads. A file lock — not a named
        // Mutex — is used deliberately: installation awaits an async download, and the
        // continuation may resume on a different thread. Mutex has thread affinity and
        // throws on release from a non-owning thread (which broke install on every path);
        // a FileStream lock is thread-agnostic and disposes safely on any thread.
        var lockPath = GetInstallLockPath(runtime.Id, rid);
        using var installLock = await AcquireInstallLockAsync(
            lockPath, TimeSpan.FromMinutes(10), cancellationToken).ConfigureAwait(false);

        // Double-check after acquiring the lock
        if (IsInstalled(runtime))
        {
            progress?.Report(new RuntimeDownloadProgress
            {
                Phase = DownloadPhase.Complete,
                Message = $"{runtime.DisplayName} is already installed"
            });
            return;
        }

        Directory.CreateDirectory(runtimeDir);

        // Phase 1: Resolve NuGet package URL
        progress?.Report(new RuntimeDownloadProgress
        {
            Phase = DownloadPhase.Resolving,
            Message = $"Resolving {packageId} v{runtime.Version}..."
        });

        var nupkgUrl = $"https://api.nuget.org/v3-flatcontainer/{packageId.ToLowerInvariant()}/{runtime.Version}/{packageId.ToLowerInvariant()}.{runtime.Version}.nupkg";

        // Phase 2: Download
        progress?.Report(new RuntimeDownloadProgress
        {
            Phase = DownloadPhase.Downloading,
            Message = $"Downloading {packageId} (~{runtime.ApproximateSizeMB}MB)..."
        });

        var tempFile = Path.Combine(Path.GetTempPath(), $"mloop-{runtime.Id}-{rid}-{Guid.NewGuid():N}.nupkg");
        try
        {
            await DownloadFileAsync(nupkgUrl, tempFile, progress, cancellationToken).ConfigureAwait(false);

            // Phase 3: Extract native libraries
            progress?.Report(new RuntimeDownloadProgress
            {
                Phase = DownloadPhase.Extracting,
                Message = "Extracting native libraries..."
            });

            var searchPath = runtime.NativeSearchPaths.GetValueOrDefault(rid, $"runtimes/{rid}/native");
            ExtractNativeLibraries(tempFile, runtimeDir, searchPath);

            // Phase 4: Verify
            progress?.Report(new RuntimeDownloadProgress
            {
                Phase = DownloadPhase.Verifying,
                Message = "Verifying installation..."
            });

            if (!IsInstalled(runtime))
            {
                // Try extracting from root (some packages don't use runtimes/ structure)
                ExtractNativeLibraries(tempFile, runtimeDir, "");
            }

            if (!IsInstalled(runtime))
                throw new InvalidOperationException($"Installation verification failed. Expected native libraries not found in {runtimeDir}");

            // Write manifest
            var manifest = $"{runtime.Id}\n{runtime.Version}\n{rid}\n{DateTime.UtcNow:O}";
            await File.WriteAllTextAsync(Path.Combine(runtimeDir, "manifest.txt"), manifest, cancellationToken).ConfigureAwait(false);

            progress?.Report(new RuntimeDownloadProgress
            {
                Phase = DownloadPhase.Complete,
                Message = $"{runtime.DisplayName} installed successfully"
            });
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    /// <summary>
    /// Acquires a cross-process install lock by opening a lock file exclusively. Unlike a
    /// named <see cref="Mutex"/>, a <see cref="FileStream"/> lock has no thread affinity, so
    /// it can be safely released after an <c>await</c> resumes on a different thread.
    /// Polls until the lock is free or <paramref name="timeout"/> elapses.
    /// </summary>
    private static async Task<FileStream> AcquireInstallLockAsync(
        string lockPath, TimeSpan timeout, CancellationToken cancellationToken)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                return new FileStream(lockPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
            }
            catch (IOException)
            {
                if (DateTime.UtcNow >= deadline)
                    throw new TimeoutException(
                        "Timed out waiting for another process to finish installing the runtime.");
                await Task.Delay(250, cancellationToken).ConfigureAwait(false);
            }
        }
    }

    private static string GetInstallLockPath(string runtimeId, string rid) =>
        Path.Combine(Path.GetTempPath(), $"mloop-runtime-{runtimeId}-{rid}.lock");

    /// <summary>
    /// Removes an installed runtime
    /// </summary>
    public void Remove(RuntimeDefinition runtime)
    {
        var rid = PlatformDetector.GetRuntimeIdentifier();
        var runtimeDir = GetRuntimeDir(runtime, rid);
        if (Directory.Exists(runtimeDir))
            Directory.Delete(runtimeDir, recursive: true);
    }

    /// <summary>
    /// Ensures the native runtime required by the given task (if any) is installed and loaded.
    /// </summary>
    /// <remarks>
    /// No-op for tasks that need no on-demand native runtime (e.g. tabular tasks). For DL tasks
    /// (image-classification, object-detection, NLP) this must be called <b>before any</b>
    /// <c>MLContext.Model.Load</c> that deserializes native model parameters — both the training
    /// path and every inference/evaluation/serving path. Skipping it on the load side surfaces as a
    /// <see cref="DllNotFoundException"/> ("Unable to load DLL 'tensorflow'/...") at deserialization
    /// time, since ML.NET resolves the native library while reconstructing the model. (BUG-40)
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// The task requires a runtime that is not yet installed (includes an install hint).
    /// </exception>
    public static void EnsureRuntimeForTask(string taskType)
    {
        if (string.IsNullOrWhiteSpace(taskType))
            return;

        var runtime = RuntimeRegistry.GetRequiredByTask(taskType);
        if (runtime == null)
            return;

        var manager = new RuntimeManager();
        if (!manager.IsInstalled(runtime))
            throw new InvalidOperationException(
                $"Task '{taskType}' requires {runtime.DisplayName} runtime (~{runtime.ApproximateSizeMB}MB). " +
                $"Install it with: mloop runtime install {runtime.Id}");

        manager.EnsureLoaded(runtime);
    }

    /// <summary>
    /// Ensures runtime is loaded for use. Call before using DL features.
    /// </summary>
    public void EnsureLoaded(RuntimeDefinition runtime)
    {
        if (_loadedRuntimes.Contains(runtime.Id))
            return;

        var rid = PlatformDetector.GetRuntimeIdentifier();
        var runtimeDir = GetRuntimeDir(runtime, rid);

        if (!IsInstalled(runtime))
            throw new InvalidOperationException(
                $"{runtime.DisplayName} is not installed. Run: mloop runtime install {runtime.Id}");

        // Register native library resolver
        if (!runtime.NativeLibraries.TryGetValue(rid, out var libs))
            return;

        // Two directories must be searchable: the runtime cache (the downloaded libtorch /
        // tensorflow natives) and the application's runtimes/<rid>/native folder, which is
        // where the managed ML stack's own native wrapper ships (e.g. TorchSharp's
        // LibTorchSharp.dll). TorchSharp's loader fails if the wrapper isn't found there.
        var appNativeDir = Path.Combine(AppContext.BaseDirectory, "runtimes", rid, "native");
        var searchDirs = new List<string> { runtimeDir };
        if (Directory.Exists(appNativeDir))
            searchDirs.Add(appNativeDir);

        if (OperatingSystem.IsWindows())
        {
            SetDllDirectory(runtimeDir);
            var path = Environment.GetEnvironmentVariable("PATH") ?? "";
            var toPrepend = searchDirs.Where(d => !path.Contains(d)).ToList();
            if (toPrepend.Count > 0)
                Environment.SetEnvironmentVariable("PATH", $"{string.Join(';', toPrepend)};{path}");
        }
        else
        {
            var ldPath = Environment.GetEnvironmentVariable("LD_LIBRARY_PATH") ?? "";
            var toPrepend = searchDirs.Where(d => !ldPath.Contains(d)).ToList();
            if (toPrepend.Count > 0)
                Environment.SetEnvironmentVariable("LD_LIBRARY_PATH", $"{string.Join(':', toPrepend)}:{ldPath}");
        }

        // Preload the downloaded native libraries (their transitive deps resolve via the
        // search path set above). Skip zero-byte stub entries some packages ship.
        foreach (var lib in libs)
        {
            var libPath = Path.Combine(runtimeDir, lib);
            if (File.Exists(libPath) && new FileInfo(libPath).Length > 0)
                NativeLibrary.TryLoad(libPath, out _);
        }

        // Preload the managed stack's native wrapper (if any) so its static initializer finds
        // it already resident — TorchSharp searches relative to its own assembly, not the cache.
        if (Directory.Exists(appNativeDir))
        {
            foreach (var wrapper in Directory.EnumerateFiles(appNativeDir))
                NativeLibrary.TryLoad(wrapper, out _);
        }

        _loadedRuntimes.Add(runtime.Id);
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetDllDirectory(string lpPathName);

    private string GetRuntimeDir(RuntimeDefinition runtime, string rid) =>
        Path.Combine(_cacheDir, runtime.Id, rid);

    private async Task DownloadFileAsync(
        string url,
        string destPath,
        IProgress<RuntimeDownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        using var response = await _httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var totalBytes = response.Content.Headers.ContentLength ?? -1;
        long downloaded = 0;

        await using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        await using var fileStream = new FileStream(destPath, FileMode.Create, FileAccess.Write, FileShare.None, 81920);

        var buffer = new byte[81920];
        int bytesRead;
        var lastReport = DateTime.UtcNow;

        while ((bytesRead = await contentStream.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
        {
            await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cancellationToken).ConfigureAwait(false);
            downloaded += bytesRead;

            if ((DateTime.UtcNow - lastReport).TotalMilliseconds > 500)
            {
                progress?.Report(new RuntimeDownloadProgress
                {
                    Phase = DownloadPhase.Downloading,
                    BytesDownloaded = downloaded,
                    TotalBytes = totalBytes,
                    Message = totalBytes > 0
                        ? $"Downloading... {downloaded / (1024 * 1024)}MB / {totalBytes / (1024 * 1024)}MB"
                        : $"Downloading... {downloaded / (1024 * 1024)}MB"
                });
                lastReport = DateTime.UtcNow;
            }
        }
    }

    private static void ExtractNativeLibraries(string nupkgPath, string destDir, string searchPath)
    {
        using var archive = ZipFile.OpenRead(nupkgPath);
        var normalizedSearch = searchPath.Replace('\\', '/').TrimEnd('/');

        foreach (var entry in archive.Entries)
        {
            if (entry.Length == 0) continue; // Skip directories

            var entryPath = entry.FullName.Replace('\\', '/');

            bool shouldExtract;
            string destName;

            if (string.IsNullOrEmpty(normalizedSearch))
            {
                // Extract native-looking files from root
                var ext = Path.GetExtension(entry.Name).ToLowerInvariant();
                shouldExtract = ext is ".dll" or ".so" or ".dylib";
                destName = entry.Name;
            }
            else
            {
                // Extract from specific path
                shouldExtract = entryPath.StartsWith(normalizedSearch + "/", StringComparison.OrdinalIgnoreCase);
                destName = shouldExtract ? entryPath[(normalizedSearch.Length + 1)..] : entry.Name;

                // Handle nested directories within search path
                if (destName.Contains('/'))
                    destName = Path.GetFileName(destName);
            }

            if (shouldExtract && !string.IsNullOrEmpty(destName))
            {
                var destPath = Path.Combine(destDir, destName);
                entry.ExtractToFile(destPath, overwrite: true);
            }
        }
    }
}

public class RuntimeStatus
{
    public required RuntimeDefinition Runtime { get; init; }
    public required bool Installed { get; init; }
    public required string Path { get; init; }
    public required long SizeBytes { get; init; }
    public required string RuntimeIdentifier { get; init; }
}

public class RuntimeDownloadProgress
{
    public DownloadPhase Phase { get; init; }
    public string Message { get; init; } = "";
    public long BytesDownloaded { get; init; }
    public long TotalBytes { get; init; }
    public double PercentComplete => TotalBytes > 0 ? (double)BytesDownloaded / TotalBytes * 100 : 0;
}

public enum DownloadPhase
{
    Resolving,
    Downloading,
    Extracting,
    Verifying,
    Complete
}
