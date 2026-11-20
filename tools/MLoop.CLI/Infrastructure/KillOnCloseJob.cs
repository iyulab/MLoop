using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace MLoop.CLI.Infrastructure;

/// <summary>
/// Ties a child process to this one on Windows: the operating system ends the child — and anything it started — when the last handle
/// to the job it was put in closes, which happens when this process dies however it dies.
/// </summary>
/// <remarks>
/// <c>mloop serve</c> runs the server as a child it waits for. <see cref="StopSignalForwarding"/> ends the child when the parent is asked
/// to stop, but a parent that is killed outright (<c>taskkill /F</c>, a supervisor's <c>Stop-Process</c>) runs nothing, so the server stayed
/// behind holding its port and its model files. A job object with <c>JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE</c> needs no cooperation from the
/// dying process: the handle is closed by the system. Unix has no equivalent a child can be given from outside; there the foreground server
/// ends with its terminal's group, and a supervisor's stop is forwarded.
/// </remarks>
[SupportedOSPlatform("windows")]
public sealed class KillOnCloseJob : IDisposable
{
    private IntPtr _job;

    private KillOnCloseJob(IntPtr job) => _job = job;

    /// <summary>
    /// Puts <paramref name="child"/> in a job that ends it when this process closes its handle. Null when the system refuses (a job the
    /// child already belongs to that does not allow nesting, which Windows before 8 does not) — the child then behaves as it did without one.
    /// </summary>
    public static KillOnCloseJob? Hold(Process child)
    {
        ArgumentNullException.ThrowIfNull(child);
        var job = CreateJobObjectW(IntPtr.Zero, null);
        if (job == IntPtr.Zero)
            return null;

        var limits = new JobObjectExtendedLimitInformation
        {
            BasicLimitInformation = new JobObjectBasicLimitInformation { LimitFlags = KillOnJobClose },
        };
        var size = Marshal.SizeOf<JobObjectExtendedLimitInformation>();
        var buffer = Marshal.AllocHGlobal(size);
        try
        {
            Marshal.StructureToPtr(limits, buffer, fDeleteOld: false);
            if (!SetInformationJobObject(job, ExtendedLimitInformation, buffer, (uint)size)
                || !AssignProcessToJobObject(job, child.Handle))
            {
                CloseHandle(job);
                return null;
            }
        }
        catch (InvalidOperationException)
        {
            // The child ended before it could be put in the job: nothing is left to hold.
            CloseHandle(job);
            return null;
        }
        finally
        {
            Marshal.FreeHGlobal(buffer);
        }

        return new KillOnCloseJob(job);
    }

    /// <summary>
    /// Closes the handle, which ends whatever is still in the job. When the child has ended on its own the job holds nothing and this is only
    /// tidiness; a process that dies never gets here and the system closes the handle for it.
    /// </summary>
    public void Dispose()
    {
        var job = Interlocked.Exchange(ref _job, IntPtr.Zero);
        if (job != IntPtr.Zero)
            CloseHandle(job);
    }

    private const int ExtendedLimitInformation = 9;           // JobObjectExtendedLimitInformation
    private const uint KillOnJobClose = 0x00002000;           // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

    [StructLayout(LayoutKind.Sequential)]
    private struct JobObjectBasicLimitInformation
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IoCounters
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JobObjectExtendedLimitInformation
    {
        public JobObjectBasicLimitInformation BasicLimitInformation;
        public IoCounters IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern IntPtr CreateJobObjectW(IntPtr attributes, string? name);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetInformationJobObject(IntPtr job, int informationClass, IntPtr information, uint length);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CloseHandle(IntPtr handle);
}
