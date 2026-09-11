using MLoop.CLI;
using Spectre.Console;

namespace MLoop.Tests;

/// <summary>
/// Drives a command line through the same dispatch <c>Main</c> uses and hands back what it wrote.
/// </summary>
/// <remarks>
/// <para>
/// Fourteen test classes carried their own copy of this — the same three rebindings (stdout, stderr,
/// Spectre's ambient console) written out each time, in two shapes that differed only in whether
/// stderr was captured. A copy is how a test-infrastructure fix stops being a fix: the next class is
/// written by copying a neighbour, and a change to the seam (a new stream to capture, a scope to
/// reset) has to be found fourteen times. One helper, one place.
/// </para>
/// <para>
/// <see cref="Console.SetOut"/> alone does not reach Spectre's ambient renderer, so the
/// <see cref="AnsiConsole.Console"/> is rebound to the same buffer — a command in <c>--json</c> mode
/// then re-routes it to stderr itself through <c>JsonOutputScope</c>, which is exactly the behaviour
/// these tests exist to observe. Stderr is always captured; a caller that does not care discards it.
/// </para>
/// <para>
/// Process-global state is rebound and restored, which is why the assembly runs its tests serially
/// (see <c>ParallelGlobalStateContractTests</c>); this helper is safe only under that arrangement.
/// </para>
/// </remarks>
internal static class CliRunner
{
    public static async Task<(int ExitCode, string Stdout, string Stderr)> RunAsync(params string[] args)
    {
        var originalOut = Console.Out;
        var originalError = Console.Error;
        var originalAnsiConsole = AnsiConsole.Console;
        var buffer = new StringWriter();
        var errorBuffer = new StringWriter();
        try
        {
            Console.SetOut(buffer);
            Console.SetError(errorBuffer);
            AnsiConsole.Console = AnsiConsole.Create(new AnsiConsoleSettings { Out = new AnsiConsoleOutput(buffer) });
            var exitCode = await Program.ExecuteAsync(Program.BuildRootCommand(), args);
            return (exitCode, buffer.ToString(), errorBuffer.ToString());
        }
        finally
        {
            Console.SetOut(originalOut);
            Console.SetError(originalError);
            AnsiConsole.Console = originalAnsiConsole;
        }
    }
}
