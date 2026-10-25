using System;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Threading.Tasks;
using MLoop.CLI.Infrastructure;
using Xunit;

namespace MLoop.Tests.Infrastructure;

/// <summary>
/// <c>mloop serve</c> says a server started only once it answers: a port another process holds is refused before
/// starting — in the moments the new server lives there, the other one answers in its name — and a server that exits
/// while binding is not reported as running.
/// </summary>
public class ServerReadinessTests
{
    [Fact]
    public void APortAnotherProcessHolds_IsInUse_AndFreeOnceReleased()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        try
        {
            Assert.True(ServerReadiness.PortInUse("localhost", port));
            Assert.True(ServerReadiness.PortInUse("127.0.0.1", port));
        }
        finally
        {
            listener.Stop();
        }
        Assert.False(ServerReadiness.PortInUse("127.0.0.1", port));
    }

    [Fact]
    public async Task AServerThatExitsBeforeAnswering_IsNotReady()
    {
        using var exited = Process.Start(new ProcessStartInfo("dotnet", "--version") { RedirectStandardOutput = true })!;
        await exited.WaitForExitAsync();

        Assert.False(await ServerReadiness.WaitUntilAnsweringAsync(exited, "127.0.0.1", FreePort(), TimeSpan.FromSeconds(5)));
    }

    [Fact]
    public async Task AServerAnsweringHealth_IsReady()
    {
        var port = FreePort();
        using var http = new HttpListener();
        http.Prefixes.Add($"http://127.0.0.1:{port}/");
        http.Start();
        var serving = Task.Run(async () =>
        {
            var context = await http.GetContextAsync();
            context.Response.StatusCode = 200;
            context.Response.Close();
        });

        Assert.True(await ServerReadiness.WaitUntilAnsweringAsync(Process.GetCurrentProcess(), "127.0.0.1", port, TimeSpan.FromSeconds(10)));
        await serving;
    }

    [Theory]
    [InlineData("*", "localhost")]
    [InlineData("0.0.0.0", "localhost")]
    [InlineData("::", "localhost")]
    [InlineData("[::1]", "[::1]")]
    [InlineData("::1", "[::1]")]
    [InlineData("127.0.0.1", "127.0.0.1")]
    [InlineData("localhost", "localhost")]
    public void AnyAddressBinds_AreProbedOnTheLoopback(string host, string probe)
        => Assert.Equal(probe, ServerReadiness.ProbeHost(host));

    private static int FreePort()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }
}
