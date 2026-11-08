using System.Net;
using System.Net.Sockets;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>mloop serve</c> refuses before it starts a server: a host that is not one, a directory that is not a project, and a port
/// something already answers on. The pieces behind the first and last had tests; the command's use of them did not, and the
/// second was reported with advice for another problem (look for MLOOP_API_PATH) to someone in the wrong directory.
/// No case starts a server: each stops before the process is created.
/// </summary>
[Collection("FileSystem")]
public sealed class ServeCommandRefusalTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string? _originalApiPath = Environment.GetEnvironmentVariable("MLOOP_API_PATH");
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-serve-" + Guid.NewGuid().ToString("N"));

    public ServeCommandRefusalTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("MLOOP_API_PATH", _originalApiPath);
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    [Theory]
    [InlineData("a b")]
    [InlineData("localhost --urls http://evil:1")]
    public async Task AHostThatIsNotAHost_IsRefusedBeforeAnythingIsLoaded(string host)
    {
        var (exit, stdout, stderr) = await CliRunner.RunAsync("serve", "--host", host);

        Assert.Equal(1, exit);
        Assert.Contains("is not a host to bind to", stdout + stderr);
        Assert.Contains("0.0.0.0", stdout + stderr);
    }

    [Fact]
    public async Task OutsideAProject_ServeSaysSo_NotThatTheApiAssemblyMightBeMissing()
    {
        var (exit, stdout, stderr) = await CliRunner.RunAsync("serve");

        Assert.Equal(1, exit);
        Assert.Contains("Not inside a MLoop project", stdout + stderr);
        Assert.Contains("mloop init", stdout + stderr);
        Assert.DoesNotContain("MLOOP_API_PATH", stdout + stderr);
    }

    [Fact]
    public async Task APortSomethingAnswersOn_IsRefused_NamingTheNextPort_AndNoServerIsStarted()
    {
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        // An assembly that exists (the command only needs to find a file) and a port that is held: the command must stop at the port.
        var fakeApi = Path.Combine(_workspace, "MLoop.API.dll");
        await File.WriteAllTextAsync(fakeApi, "not an assembly");
        Environment.SetEnvironmentVariable("MLOOP_API_PATH", fakeApi);
        using var holder = new TcpListener(IPAddress.Loopback, 0);
        holder.Start();
        var port = ((IPEndPoint)holder.LocalEndpoint).Port;

        var (exit, stdout, stderr) = await CliRunner.RunAsync("serve", "--host", "127.0.0.1", "--port", port.ToString());

        Assert.Equal(1, exit);
        Assert.Contains($"Port {port} is already in use on 127.0.0.1", stdout + stderr);
        Assert.Contains($"--port {port + 1}", stdout + stderr);
        Assert.DoesNotContain("Server started", stdout + stderr);
    }
}
