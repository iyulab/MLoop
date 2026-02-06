using System.Runtime.InteropServices;

namespace MLoop.CLI.Infrastructure.Update;

public enum InstallMethod
{
    DotnetTool,
    StandaloneBinary
}

public static class InstallDetector
{
    public static InstallMethod Detect()
    {
        var processPath = Environment.ProcessPath;
        if (string.IsNullOrEmpty(processPath))
            return InstallMethod.StandaloneBinary;

        // dotnet tool installs to paths containing .dotnet/tools
        var normalizedPath = processPath.Replace('\\', '/');
        if (normalizedPath.Contains(".dotnet/tools", StringComparison.OrdinalIgnoreCase))
            return InstallMethod.DotnetTool;

        return InstallMethod.StandaloneBinary;
    }

    public static string GetRuntimeIdentifier()
    {
        var os = OperatingSystem.IsWindows() ? "win"
            : OperatingSystem.IsMacOS() ? "osx"
            : "linux";

        var arch = RuntimeInformation.OSArchitecture switch
        {
            Architecture.X64 => "x64",
            Architecture.Arm64 => "arm64",
            _ => "x64"
        };

        return $"{os}-{arch}";
    }
}
