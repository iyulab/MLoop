using Spectre.Console;

namespace MLoop.CLI.Infrastructure.Diagnostics;

/// <summary>
/// Reserves stdout for machine-readable output: while this scope is active, stdout receives nothing
/// except what the caller writes to <see cref="Stdout"/>.
/// </summary>
/// <remarks>
/// <para>
/// A command with a <c>--json</c> mode cannot keep that promise call site by call site.
/// <c>mloop train</c> alone renders through Spectre in roughly sixty places and narrates through
/// <c>Console.WriteLine</c> from three assemblies, and ML.NET writes on its own account — one missed
/// line makes the whole stream unparseable for the consumer. So the two default sinks are replaced
/// once, here: plain <see cref="Console.Out"/> and the ambient <see cref="AnsiConsole"/> both go to
/// <see cref="TextWriter.Null"/>, and the real stdout is handed back to the caller as
/// <see cref="Stdout"/>.
/// </para>
/// <para>
/// Diagnostics are unaffected: <see cref="ErrorConsole"/> binds to <see cref="Console.Error"/> on
/// each access and never reads the ambient console, so the "<c>exit != 0</c> ⇒ stderr is not empty"
/// contract still holds inside this scope. Narration is discarded rather than moved to stderr on
/// purpose — a successful run leaving output on stderr would break the other half of that same
/// convention for consumers who treat any stderr as a failure signal.
/// </para>
/// </remarks>
public sealed class MachineOutputScope : IDisposable
{
    private readonly TextWriter _previousOut;
    private readonly IAnsiConsole _previousConsole;

    /// <summary>
    /// The active scope, or null outside machine mode. Static because a CLI process runs exactly one
    /// command: this is what lets the diagnostics sink report a failure into the event stream without
    /// every early return in every command having to know about it.
    /// </summary>
    public static MachineOutputScope? Current { get; private set; }

    public MachineOutputScope()
    {
        _previousOut = Console.Out;
        _previousConsole = AnsiConsole.Console;
        Stdout = _previousOut;

        Console.SetOut(TextWriter.Null);
        AnsiConsole.Console = AnsiConsole.Create(new AnsiConsoleSettings
        {
            Out = new AnsiConsoleOutput(TextWriter.Null)
        });

        Current = this;
    }

    /// <summary>The real stdout, for the machine-readable stream this scope exists to protect.</summary>
    public TextWriter Stdout { get; }

    /// <summary>
    /// Where a failure written to stderr is also reported as an event. Set by the command that owns
    /// the stream; null until then, so a scope without an emitter simply stays silent.
    /// </summary>
    public Action<string>? ErrorSink { get; set; }

    /// <summary>
    /// Where a warning raised through <see cref="WarningConsole"/> is reported as an event. Same
    /// ownership rule as <see cref="ErrorSink"/>: null until the command wires its emitter.
    /// </summary>
    public Action<string>? WarningSink { get; set; }

    /// <summary>
    /// Called by the stderr diagnostics sinks. Markup is stripped — the event carries the message a
    /// consumer would print, not Spectre's rendering instructions.
    /// </summary>
    public static void ReportError(string message) => Report(Current?.ErrorSink, message);

    /// <summary>Called by <see cref="WarningConsole"/>. Markup is stripped, as for errors.</summary>
    public static void ReportWarning(string message) => Report(Current?.WarningSink, message);

    private static void Report(Action<string>? sink, string message)
    {
        if (sink is null)
            return;

        var plain = message;
        try { plain = Markup.Remove(message); }
        catch (InvalidOperationException) { /* not valid markup — report it verbatim */ }

        sink(plain);
    }

    public void Dispose()
    {
        Console.SetOut(_previousOut);
        AnsiConsole.Console = _previousConsole;
        if (ReferenceEquals(Current, this))
            Current = null;
    }
}
