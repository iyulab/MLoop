using System.Net;
using System.Text;
using MLoop.CLI.Commands;
using MLoop.CLI.Infrastructure.Update;

namespace MLoop.Tests.Infrastructure.Update;

/// <summary>
/// The update check, the binary download and <c>mloop update</c> itself, without a network: a stand-in handler answers the two
/// requests (the releases list, the asset), so the success path and every way the network can fail run on any machine. The cache
/// directory, the executable to replace and the install method are pointed at scratch values, so no test touches the user's own.
/// </summary>
public sealed class UpdateNetworkTests : IDisposable
{
    private static readonly string Rid = InstallDetector.GetRuntimeIdentifier();
    private static readonly string Asset = $"mloop-{Rid}{(Rid.StartsWith("win", StringComparison.Ordinal) ? ".exe" : "")}";
    private const string ReleasesUrlPart = "api.github.com/repos/iyulab/MLoop/releases";

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "mloop_update_" + Guid.NewGuid().ToString("N"));
    private readonly HttpMessageHandler _handler0 = UpdateChecker.Handler;
    private readonly string _cache0 = UpdateChecker.CacheDirectory;
    private readonly string? _exe0 = UpdateChecker.ExecutablePath;
    private readonly Func<InstallMethod> _detect0 = UpdateCommand.DetectInstall;
    private readonly (string, string) _tool0 = UpdateCommand.ToolUpdateCommand;
    private readonly List<string> _requests = [];

    public UpdateNetworkTests()
    {
        Directory.CreateDirectory(_scratch);
        UpdateChecker.CacheDirectory = Path.Combine(_scratch, "cache");
    }

    public void Dispose()
    {
        UpdateChecker.Handler = _handler0;
        UpdateChecker.CacheDirectory = _cache0;
        UpdateChecker.ExecutablePath = _exe0;
        UpdateCommand.DetectInstall = _detect0;
        UpdateCommand.ToolUpdateCommand = _tool0;
        try { Directory.Delete(_scratch, recursive: true); } catch (IOException) { }
    }

    private sealed class Stub(Func<HttpRequestMessage, HttpResponseMessage> answer) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => Task.FromResult(answer(request));
    }

    private void Serve(Func<HttpRequestMessage, HttpResponseMessage> answer)
        => UpdateChecker.Handler = new Stub(request =>
        {
            _requests.Add(request.RequestUri!.ToString());
            return answer(request);
        });

    private static HttpResponseMessage Json(string body) => new(HttpStatusCode.OK) { Content = new StringContent(body, Encoding.UTF8, "application/json") };

    private static string Releases(params (string Tag, string[] Assets)[] releases)
        => "[" + string.Join(",", releases.Select(r =>
            $"{{\"tag_name\":\"{r.Tag}\",\"assets\":[{string.Join(",", r.Assets.Select(a => $"{{\"name\":\"{a}\"}}"))}]}}")) + "]";

    private void ServeReleases(string releasesJson, byte[]? binary = null)
        => Serve(request => request.RequestUri!.ToString().Contains(ReleasesUrlPart, StringComparison.Ordinal)
            ? Json(releasesJson)
            : binary is null
                ? new HttpResponseMessage(HttpStatusCode.NotFound)
                : new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(binary) });

    // --- CheckForUpdateAsync ---

    [Fact]
    public async Task ANewerRelease_WithThisPlatformsBinary_IsAnAvailableUpdate_AndTheAnswerIsCached()
    {
        ServeReleases(Releases(("v9.9.9", [Asset])));

        var info = await UpdateChecker.CheckForUpdateAsync(forceCheck: true);

        Assert.NotNull(info);
        Assert.Equal("9.9.9", info.LatestVersion);
        Assert.True(info.UpdateAvailable);
        Assert.True(File.Exists(Path.Combine(UpdateChecker.CacheDirectory, "update-check.json")));
    }

    [Fact]
    public async Task TheLatestReleaseWithThisPlatformsBinary_Wins_AReleaseWithoutOneIsSkipped_EvenWhenNewer()
    {
        ServeReleases(Releases(("v99.0.0", ["mloop-some-other-platform"]), ("v9.0.0", [Asset]), ("v8.0.0", [Asset])));

        var info = await UpdateChecker.CheckForUpdateAsync(forceCheck: true);

        Assert.Equal("9.0.0", info!.LatestVersion);
    }

    [Fact]
    public async Task ReleasesThatNoneCarryThisPlatformsBinary_AreNoAnswer()
    {
        ServeReleases(Releases(("v9.0.0", ["mloop-some-other-platform"])));

        Assert.Null(await UpdateChecker.CheckForUpdateAsync(forceCheck: true));
    }

    [Fact]
    public async Task ARunningVersion_AtOrAboveTheLatestRelease_HasNoUpdateAvailable()
    {
        ServeReleases(Releases(("v0.0.1", [Asset])));

        var info = await UpdateChecker.CheckForUpdateAsync(forceCheck: true);

        Assert.False(info!.UpdateAvailable);
    }

    [Theory]
    [InlineData("[]")]
    [InlineData("not json")]
    public async Task AnEmptyOrUnreadableReleaseList_IsNoAnswer_NeverAnException(string body)
    {
        Serve(_ => Json(body));

        Assert.Null(await UpdateChecker.CheckForUpdateAsync(forceCheck: true));
    }

    [Fact]
    public async Task ANetworkFailure_IsNoAnswer_NeverAnException()
    {
        UpdateChecker.Handler = new Stub(_ => throw new HttpRequestException("no route to host"));

        Assert.Null(await UpdateChecker.CheckForUpdateAsync(forceCheck: true));
    }

    [Fact]
    public async Task AFreshCachedAnswer_IsUsedWithoutAskingTheNetwork_AStaleOne_IsNot()
    {
        ServeReleases(Releases(("v9.9.9", [Asset])));
        await UpdateChecker.CheckForUpdateAsync(forceCheck: true);
        _requests.Clear();

        var cached = await UpdateChecker.CheckForUpdateAsync();
        Assert.Equal("9.9.9", cached!.LatestVersion);
        Assert.Empty(_requests);

        var cacheFile = Path.Combine(UpdateChecker.CacheDirectory, "update-check.json");
        File.WriteAllText(cacheFile, $"{{\"latestVersion\":\"1.0.0\",\"checkedAt\":\"{DateTime.UtcNow.AddHours(-25):O}\"}}");
        var refreshed = await UpdateChecker.CheckForUpdateAsync();
        Assert.Equal("9.9.9", refreshed!.LatestVersion);
        Assert.Single(_requests);
    }

    [Fact]
    public async Task AnUnreadableCache_IsIgnored_TheNetworkAnswers()
    {
        Directory.CreateDirectory(UpdateChecker.CacheDirectory);
        File.WriteAllText(Path.Combine(UpdateChecker.CacheDirectory, "update-check.json"), "{ not json");
        ServeReleases(Releases(("v9.9.9", [Asset])));

        Assert.Equal("9.9.9", (await UpdateChecker.CheckForUpdateAsync())!.LatestVersion);
    }

    // --- DownloadLatestBinaryAsync ---

    [Fact]
    public async Task TheDownload_AsksForTheVersionsAssetOfThisPlatform_WritesIt_AndReportsProgressToTheTotal()
    {
        var binary = Enumerable.Range(0, 200_000).Select(i => (byte)(i % 251)).ToArray();
        ServeReleases("[]", binary);
        var destination = Path.Combine(_scratch, "downloaded");
        var progress = new List<(long Downloaded, long? Total)>();

        await UpdateChecker.DownloadLatestBinaryAsync("9.9.9", Rid, destination, (done, total) => progress.Add((done, total)));

        Assert.Equal(binary, await File.ReadAllBytesAsync(destination));
        Assert.EndsWith($"/releases/download/v9.9.9/{Asset}", _requests.Single(), StringComparison.Ordinal);
        Assert.Equal(binary.Length, progress[^1].Downloaded);
        Assert.Equal(binary.Length, progress[^1].Total);
        Assert.True(progress.Count > 1, "a 200 KB body arrives in more than one read");
    }

    [Fact]
    public async Task ADownloadTheServerRefuses_Throws_AndLeavesNoFileBehind()
    {
        ServeReleases("[]", binary: null);   // the asset answers 404
        var destination = Path.Combine(_scratch, "downloaded");

        await Assert.ThrowsAsync<HttpRequestException>(() => UpdateChecker.DownloadLatestBinaryAsync("9.9.9", Rid, destination));

        Assert.False(File.Exists(destination));
    }

    // --- ReplaceExecutable ---

    [Fact]
    public void OnUnix_TheExecutableIsOverwritten_TheDownloadIsConsumed_AndTheResultCanRun()
    {
        Assert.SkipWhen(OperatingSystem.IsWindows(), "the Windows path renames instead (tested below)");
        var current = Path.Combine(_scratch, "mloop");
        var download = Path.Combine(_scratch, "mloop-update");
        File.WriteAllText(current, "old");
        File.WriteAllText(download, "new");
        UpdateChecker.ExecutablePath = current;

        UpdateChecker.ReplaceExecutable(download);

        Assert.Equal("new", File.ReadAllText(current));
        Assert.False(File.Exists(download));
        if (!OperatingSystem.IsWindows())
            Assert.True(File.GetUnixFileMode(current).HasFlag(UnixFileMode.UserExecute));
    }

    [Fact]
    public void OnWindows_TheRunningExecutableIsMovedAside_AndTheDownloadTakesItsPlace()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "a running executable can only be renamed on Windows");
        var current = Path.Combine(_scratch, "mloop.exe");
        var download = Path.Combine(_scratch, "mloop-update.exe");
        File.WriteAllText(current, "old");
        File.WriteAllText(download, "new");
        UpdateChecker.ExecutablePath = current;

        UpdateChecker.ReplaceExecutable(download);

        Assert.Equal("new", File.ReadAllText(current));
        Assert.Equal("old", File.ReadAllText(current + ".old"));
    }

    // --- mloop update ---

    [Fact]
    public async Task Update_WhenTheVersionCannotBeChecked_ExitsNonZero()
    {
        UpdateChecker.Handler = new Stub(_ => throw new HttpRequestException("offline"));

        Assert.Equal(1, await UpdateCommand.ExecuteAsync(checkOnly: false));
    }

    [Fact]
    public async Task Update_WhenAlreadyCurrent_ExitsZero_AndInstallsNothing()
    {
        ServeReleases(Releases(("v0.0.1", [Asset])));
        UpdateCommand.DetectInstall = () => throw new InvalidOperationException("nothing to install, so nothing to detect");

        Assert.Equal(0, await UpdateCommand.ExecuteAsync(checkOnly: false));
    }

    [Fact]
    public async Task UpdateCheck_WithANewerVersion_ExitsZero_WithoutDownloading()
    {
        ServeReleases(Releases(("v9.9.9", [Asset])), binary: [1, 2, 3]);
        UpdateCommand.DetectInstall = () => throw new InvalidOperationException("--check installs nothing");

        Assert.Equal(0, await UpdateCommand.ExecuteAsync(checkOnly: true));
        Assert.DoesNotContain(_requests, r => r.Contains("/download/", StringComparison.Ordinal));
    }

    [Fact]
    public async Task Update_OfADotnetToolInstall_ExitsWithTheToolsOwnExitCode()
    {
        ServeReleases(Releases(("v9.9.9", [Asset])));
        UpdateCommand.DetectInstall = () => InstallMethod.DotnetTool;

        UpdateCommand.ToolUpdateCommand = ("dotnet", "--version");
        Assert.Equal(0, await UpdateCommand.ExecuteAsync(checkOnly: false));

        UpdateCommand.ToolUpdateCommand = ("dotnet", "a-command-dotnet-does-not-have");
        Assert.NotEqual(0, await UpdateCommand.ExecuteAsync(checkOnly: false));
    }

    [Fact]
    public async Task Update_WhenTheToolCannotBeStarted_ExitsNonZero_NotWithAStackTrace()
    {
        ServeReleases(Releases(("v9.9.9", [Asset])));
        UpdateCommand.DetectInstall = () => InstallMethod.DotnetTool;
        UpdateCommand.ToolUpdateCommand = (Path.Combine(_scratch, "no-such-program"), "update");

        Assert.Equal(1, await UpdateCommand.ExecuteAsync(checkOnly: false));
    }

    [Fact]
    public async Task Update_OfAStandaloneBinary_ReplacesTheExecutable_AndExitsZero()
    {
        var current = Path.Combine(_scratch, "mloop");
        File.WriteAllText(current, "old");
        UpdateChecker.ExecutablePath = current;
        ServeReleases(Releases(("v9.9.9", [Asset])), binary: Encoding.UTF8.GetBytes("new build"));
        UpdateCommand.DetectInstall = () => InstallMethod.StandaloneBinary;
        Assert.SkipWhen(OperatingSystem.IsWindows(), "the Windows replacement is covered by its own test");

        Assert.Equal(0, await UpdateCommand.ExecuteAsync(checkOnly: false));

        Assert.Equal("new build", File.ReadAllText(current));
    }

    [Fact]
    public async Task Update_OfAStandaloneBinary_WhenTheDownloadFails_ExitsNonZero_AndKeepsTheExecutable()
    {
        var current = Path.Combine(_scratch, "mloop");
        File.WriteAllText(current, "old");
        UpdateChecker.ExecutablePath = current;
        ServeReleases(Releases(("v9.9.9", [Asset])), binary: null);   // the asset answers 404
        UpdateCommand.DetectInstall = () => InstallMethod.StandaloneBinary;

        Assert.Equal(1, await UpdateCommand.ExecuteAsync(checkOnly: false));

        Assert.Equal("old", File.ReadAllText(current));
    }
}
