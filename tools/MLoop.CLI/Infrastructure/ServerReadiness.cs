using System.Diagnostics;
using System.Net;
using System.Net.Sockets;

namespace MLoop.CLI.Infrastructure;

/// <summary>
/// Whether a server can start on an address, and whether it has: <c>mloop serve</c> says the server started only once
/// it answers there. Starting a process is not starting a server — on a port another process already holds, the
/// server exits moments later, and in that window whatever holds the port answers in its name.
/// </summary>
internal static class ServerReadiness
{
    /// <summary>
    /// Whether <paramref name="port"/> is already taken on the addresses <paramref name="host"/> binds to. A host name
    /// that does not resolve is not judged here (false) — the server reports it.
    /// </summary>
    public static bool PortInUse(string host, int port)
    {
        foreach (var address in BindAddresses(host))
        {
            try
            {
                var listener = new TcpListener(address, port) { ExclusiveAddressUse = true };
                listener.Start();
                listener.Stop();
            }
            catch (SocketException ex) when (ex.SocketErrorCode is SocketError.AddressAlreadyInUse or SocketError.AccessDenied)
            {
                return true;
            }
            catch (SocketException)
            {
                // An address family the machine does not have (no IPv6) is not a taken port.
            }
        }
        return false;
    }

    /// <summary>
    /// Waits until the server answers <c>/health</c> on <paramref name="host"/>:<paramref name="port"/>. False when
    /// <paramref name="process"/> exits first, or when <paramref name="timeout"/> passes; the process is left running.
    /// </summary>
    public static async Task<bool> WaitUntilAnsweringAsync(Process process, string host, int port, TimeSpan timeout, CancellationToken cancellationToken = default)
    {
        using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(2) };
        var health = new Uri($"http://{ProbeHost(host)}:{port}/health");
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            if (process.HasExited)
                return false;
            try
            {
                using var response = await http.GetAsync(health, cancellationToken);
                if (response.IsSuccessStatusCode)
                    return !process.HasExited;
            }
            catch (HttpRequestException)
            {
            }
            catch (TaskCanceledException) when (!cancellationToken.IsCancellationRequested)
            {
            }
            await Task.Delay(250, cancellationToken);
        }
        return false;
    }

    // Where a client reaches a server bound to host: an any-address bind is reached on the loopback.
    internal static string ProbeHost(string host) => host.Trim('[', ']') switch
    {
        "*" or "+" or "0.0.0.0" or "::" => "localhost",
        var h when IPAddress.TryParse(h, out var ip) && ip.AddressFamily == AddressFamily.InterNetworkV6 => $"[{h}]",
        var h => h,
    };

    private static IEnumerable<IPAddress> BindAddresses(string host)
    {
        var h = host.Trim('[', ']');
        if (h is "*" or "+" or "0.0.0.0")
            return [IPAddress.Any];
        if (h == "::")
            return [IPAddress.IPv6Any];
        if (string.Equals(h, "localhost", StringComparison.OrdinalIgnoreCase))
            return [IPAddress.Loopback, IPAddress.IPv6Loopback];
        if (IPAddress.TryParse(h, out var ip))
            return [ip];
        try
        {
            return Dns.GetHostAddresses(h);
        }
        catch (SocketException)
        {
            return [];
        }
    }
}
