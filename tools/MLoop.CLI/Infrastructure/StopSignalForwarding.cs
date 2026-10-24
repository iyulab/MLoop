using System.Diagnostics;
using System.Runtime.InteropServices;

namespace MLoop.CLI.Infrastructure;

/// <summary>
/// Passes a stop request the parent receives (SIGTERM, SIGINT, SIGQUIT) on to a child process it runs in the
/// foreground, and keeps the parent waiting for the child to finish.
/// </summary>
/// <remarks>
/// <c>mloop serve</c> is a parent that runs the server as a child. A terminal's Ctrl+C reaches both — it signals the
/// whole foreground group — but a supervisor signals one process: <c>docker stop</c> sends SIGTERM to the container's
/// first process, which is <c>mloop serve</c>. The parent then exited at once and the server did not get the signal:
/// outside a container it went on running and held the port, inside one it was killed when the container stopped, its
/// graceful shutdown (requests in flight, a training job) skipped. Forwarding is SIGTERM on Unix, which the server's
/// host turns into a graceful stop; Windows has no signal to forward, so the child there is ended with its tree.
/// </remarks>
public sealed class StopSignalForwarding : IDisposable
{
    private readonly List<PosixSignalRegistration> _registrations = [];

    private StopSignalForwarding(Process child)
    {
        foreach (var signal in new[] { PosixSignal.SIGTERM, PosixSignal.SIGINT, PosixSignal.SIGQUIT })
        {
            _registrations.Add(PosixSignalRegistration.Create(signal, context =>
            {
                // The parent stays to report how the child ended; the child decides when that is.
                context.Cancel = true;
                Stop(child);
            }));
        }
    }

    /// <summary>Forwards stop requests to <paramref name="child"/> until disposed.</summary>
    public static StopSignalForwarding To(Process child) => new(child);

    internal static void Stop(Process child)
    {
        try
        {
            if (child.HasExited)
                return;
            if (OperatingSystem.IsWindows())
                child.Kill(entireProcessTree: true);
            else
                _ = kill(child.Id, SIGTERM);
        }
        catch (InvalidOperationException)
        {
            // The child is gone already.
        }
    }

    private const int SIGTERM = 15;

    [DllImport("libc", SetLastError = true)]
    private static extern int kill(int pid, int sig);

    public void Dispose()
    {
        foreach (var registration in _registrations)
            registration.Dispose();
    }
}
