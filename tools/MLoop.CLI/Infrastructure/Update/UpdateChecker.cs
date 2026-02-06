using System.Net.Http.Json;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace MLoop.CLI.Infrastructure.Update;

public record UpdateInfo(string LatestVersion, string CurrentVersion, bool UpdateAvailable);

public static class UpdateChecker
{
    private static readonly string CacheDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "mloop");

    private static readonly string CachePath = Path.Combine(CacheDir, "update-check.json");

    private const string GitHubReleasesUrl = "https://api.github.com/repos/iyulab/MLoop/releases?per_page=15";

    public static string GetCurrentVersion()
    {
        var attr = Assembly.GetExecutingAssembly()
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>();

        if (attr is null)
            return "0.0.0";

        var version = attr.InformationalVersion;

        // Remove +commitHash suffix if present
        var plusIndex = version.IndexOf('+');
        if (plusIndex >= 0)
            version = version[..plusIndex];

        return version;
    }

    public static async Task<UpdateInfo?> CheckForUpdateAsync(bool forceCheck = false)
    {
        try
        {
            // Check cache first
            if (!forceCheck)
            {
                var cached = ReadCache();
                if (cached is not null)
                    return cached;
            }

            var timeout = forceCheck ? TimeSpan.FromSeconds(30) : TimeSpan.FromSeconds(2);

            using var http = new HttpClient { Timeout = timeout };
            http.DefaultRequestHeaders.UserAgent.ParseAdd("MLoop-CLI");

            var releases = await http.GetFromJsonAsync(
                GitHubReleasesUrl, GitHubSourceGenerationContext.Default.ListGitHubRelease);

            if (releases is null || releases.Count == 0)
                return null;

            // Find the latest release that has a binary asset for our platform
            var rid = InstallDetector.GetRuntimeIdentifier();
            var ext = rid.StartsWith("win") ? ".exe" : "";
            var expectedAsset = $"mloop-{rid}{ext}";

            string? latestVersion = null;
            foreach (var release in releases)
            {
                if (release.TagName is null) continue;
                var hasAsset = release.Assets?.Exists(a => a.Name == expectedAsset) ?? false;
                if (!hasAsset) continue;

                var version = release.TagName.TrimStart('v');
                if (latestVersion is null || CompareVersions(version, latestVersion) > 0)
                    latestVersion = version;
            }

            if (latestVersion is null)
                return null;

            var currentVersion = GetCurrentVersion();
            var updateAvailable = CompareVersions(latestVersion, currentVersion) > 0;

            var info = new UpdateInfo(latestVersion, currentVersion, updateAvailable);
            WriteCache(info);
            return info;
        }
        catch
        {
            // Silently fail - update check should never break CLI
            return null;
        }
    }

    public static async Task DownloadLatestBinaryAsync(
        string version, string rid, string destinationPath, Action<long, long?>? onProgress = null)
    {
        var ext = rid.StartsWith("win") ? ".exe" : "";
        var assetName = $"mloop-{rid}{ext}";
        var downloadUrl = $"https://github.com/iyulab/MLoop/releases/download/v{version}/{assetName}";

        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("MLoop-CLI");

        using var response = await http.GetAsync(downloadUrl, HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();

        var totalBytes = response.Content.Headers.ContentLength;

        await using var contentStream = await response.Content.ReadAsStreamAsync();
        await using var fileStream = new FileStream(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None);

        var buffer = new byte[81920];
        long totalRead = 0;
        int bytesRead;

        while ((bytesRead = await contentStream.ReadAsync(buffer)) > 0)
        {
            await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead));
            totalRead += bytesRead;
            onProgress?.Invoke(totalRead, totalBytes);
        }
    }

    public static void ReplaceExecutable(string newFilePath)
    {
        var currentPath = Environment.ProcessPath
            ?? throw new InvalidOperationException("Cannot determine current executable path.");

        if (OperatingSystem.IsWindows())
        {
            // Windows: can't overwrite running exe, rename to .old first
            var oldPath = currentPath + ".old";
            if (File.Exists(oldPath))
                File.Delete(oldPath);

            File.Move(currentPath, oldPath);
            File.Move(newFilePath, currentPath);
        }
        else
        {
            // Unix: can overwrite running exe
            File.Copy(newFilePath, currentPath, overwrite: true);
            File.Delete(newFilePath);

            // chmod +x
            File.SetUnixFileMode(currentPath,
                UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute |
                UnixFileMode.GroupRead | UnixFileMode.GroupExecute |
                UnixFileMode.OtherRead | UnixFileMode.OtherExecute);
        }
    }

    public static void CleanupOldBinary()
    {
        try
        {
            if (!OperatingSystem.IsWindows()) return;

            var processPath = Environment.ProcessPath;
            if (string.IsNullOrEmpty(processPath)) return;

            var oldPath = processPath + ".old";
            if (File.Exists(oldPath))
                File.Delete(oldPath);
        }
        catch
        {
            // Best-effort cleanup
        }
    }

    /// <summary>
    /// Compare two SemVer strings. Returns positive if a > b, negative if a < b, 0 if equal.
    /// Prerelease versions are lower than release versions (e.g., 0.2.0 > 0.2.0-alpha).
    /// </summary>
    internal static int CompareVersions(string a, string b)
    {
        SplitVersion(a, out var aParts, out var aPrerelease);
        SplitVersion(b, out var bParts, out var bPrerelease);

        // Compare numeric parts
        var maxLen = Math.Max(aParts.Length, bParts.Length);
        for (int i = 0; i < maxLen; i++)
        {
            var aVal = i < aParts.Length ? aParts[i] : 0;
            var bVal = i < bParts.Length ? bParts[i] : 0;
            if (aVal != bVal)
                return aVal.CompareTo(bVal);
        }

        // Numeric parts equal - compare prerelease
        // No prerelease > has prerelease (1.0.0 > 1.0.0-alpha)
        if (string.IsNullOrEmpty(aPrerelease) && !string.IsNullOrEmpty(bPrerelease))
            return 1;
        if (!string.IsNullOrEmpty(aPrerelease) && string.IsNullOrEmpty(bPrerelease))
            return -1;

        return string.Compare(aPrerelease, bPrerelease, StringComparison.OrdinalIgnoreCase);
    }

    private static void SplitVersion(string version, out int[] numericParts, out string prerelease)
    {
        var dashIndex = version.IndexOf('-');
        string numericStr;
        if (dashIndex >= 0)
        {
            numericStr = version[..dashIndex];
            prerelease = version[(dashIndex + 1)..];
        }
        else
        {
            numericStr = version;
            prerelease = "";
        }

        numericParts = numericStr.Split('.').Select(s => int.TryParse(s, out var n) ? n : 0).ToArray();
    }

    private static UpdateInfo? ReadCache()
    {
        try
        {
            if (!File.Exists(CachePath))
                return null;

            var json = File.ReadAllText(CachePath);
            var cache = JsonSerializer.Deserialize(json, UpdateCacheContext.Default.UpdateCache);
            if (cache is null)
                return null;

            // Cache valid for 24 hours
            if (DateTime.UtcNow - cache.CheckedAt > TimeSpan.FromHours(24))
                return null;

            return new UpdateInfo(cache.LatestVersion, GetCurrentVersion(),
                CompareVersions(cache.LatestVersion, GetCurrentVersion()) > 0);
        }
        catch
        {
            return null;
        }
    }

    private static void WriteCache(UpdateInfo info)
    {
        try
        {
            Directory.CreateDirectory(CacheDir);
            var cache = new UpdateCache
            {
                LatestVersion = info.LatestVersion,
                CheckedAt = DateTime.UtcNow
            };
            var json = JsonSerializer.Serialize(cache, UpdateCacheContext.Default.UpdateCache);
            File.WriteAllText(CachePath, json);
        }
        catch
        {
            // Best-effort cache write
        }
    }
}

internal class UpdateCache
{
    [JsonPropertyName("latestVersion")]
    public string LatestVersion { get; set; } = "";

    [JsonPropertyName("checkedAt")]
    public DateTime CheckedAt { get; set; }
}

[JsonSerializable(typeof(UpdateCache))]
internal partial class UpdateCacheContext : JsonSerializerContext;

internal class GitHubRelease
{
    [JsonPropertyName("tag_name")]
    public string? TagName { get; set; }

    [JsonPropertyName("assets")]
    public List<GitHubAsset>? Assets { get; set; }
}

internal class GitHubAsset
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = "";
}

[JsonSerializable(typeof(GitHubRelease))]
[JsonSerializable(typeof(List<GitHubRelease>))]
internal partial class GitHubSourceGenerationContext : JsonSerializerContext;
