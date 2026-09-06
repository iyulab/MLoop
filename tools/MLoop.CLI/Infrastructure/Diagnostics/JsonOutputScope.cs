using Spectre.Console;

namespace MLoop.CLI.Infrastructure.Diagnostics;

/// <summary>
/// Owns stdout for a command running in <c>--json</c> mode: human-facing output is routed to stderr
/// for the scope's lifetime, and stdout is <b>guaranteed</b> to have received a document by the time
/// the scope closes.
/// </summary>
/// <remarks>
/// <para>
/// The half that routes narration to stderr was previously repeated verbatim in every structured-output
/// command. The half that matters more is the guarantee: a command owes its consumer a parseable
/// document at every exit point, and error exits were reaching <c>return 1</c> with an empty stdout —
/// which fails <c>JSON.parse</c> exactly as prose does. Reached through <c>predict</c>, which has
/// twelve such exits; guarding each one is the approach <see cref="MachineOutputScope"/> already
/// rejects for its own stream ("cannot keep that promise call site by call site"), and it breaks again
/// the moment a thirteenth is added.
/// </para>
/// <para>
/// This is deliberately a <b>guarantee, not an interception</b>. An earlier design had the diagnostics
/// sink write <c>{"error": …}</c> the moment a failure was reported, which made correctness depend on
/// call ordering: <c>info</c> reports the failure through <see cref="ErrorConsole"/> and only then
/// emits its own richer payload (a full document carrying an <c>errors</c> array), so an intercepting
/// sink would have written the poorer document first and left two on stdout. Checking at the end
/// instead — did anything reach stdout? — needs no per-command knowledge: a command that emitted its
/// own payload is already correct and this does nothing, and one that did not gets the failure it
/// reported.
/// </para>
/// <para>
/// Distinct from <see cref="MachineOutputScope"/>, which <i>discards</i> stdout writes so the owning
/// command can hand a reserved stream to its own emitter (<c>train</c>'s event stream). This one
/// forwards every write through untouched and only observes that one happened.
/// </para>
/// </remarks>
public sealed class JsonOutputScope : IDisposable
{
    private readonly TextWriter _previousOut;
    private readonly IAnsiConsole _previousConsole;
    private readonly ObservingWriter _stdout;
    private readonly JsonOutputScope? _previousScope;
    private string? _reportedError;

    /// <summary>
    /// The active scope, or null outside <c>--json</c> mode. Static for the same reason
    /// <see cref="MachineOutputScope.Current"/> is: a CLI process runs exactly one command, so the
    /// diagnostics sink can reach the scope without every call site having to know it exists.
    /// </summary>
    public static JsonOutputScope? Current { get; private set; }

    public JsonOutputScope()
    {
        _previousOut = Console.Out;
        _previousConsole = AnsiConsole.Console;
        _previousScope = Current;

        _stdout = new ObservingWriter(_previousOut);
        Console.SetOut(_stdout);
        AnsiConsole.Console = AnsiConsole.Create(new AnsiConsoleSettings
        {
            Out = new AnsiConsoleOutput(Console.Error)
        });

        Current = this;
    }

    /// <summary>
    /// Called by <see cref="ErrorConsole"/> when a failure is reported. The <b>first</b> report wins:
    /// as a stack unwinds, the innermost sink names the actual cause while the outermost handler
    /// re-reports a wrapper of it, and the specific message is the one worth handing a consumer.
    /// Markup is stripped, as for the event stream.
    /// </summary>
    public static void ReportError(string message)
    {
        var scope = Current;
        if (scope is null || scope._reportedError is not null)
            return;

        var plain = message;
        try { plain = Markup.Remove(message); }
        catch (InvalidOperationException) { /* not valid markup — keep it verbatim */ }

        scope._reportedError = plain;
    }

    public void Dispose()
    {
        Console.SetOut(_previousOut);
        AnsiConsole.Console = _previousConsole;
        if (ReferenceEquals(Current, this))
            Current = _previousScope;

        // A reported failure must reach the consumer. When the command already wrote its own
        // document — including one that describes the failure in a richer shape — this adds nothing.
        if (_reportedError is not null && !_stdout.WroteAnything)
            JsonError.Emit(_reportedError);

        // Deliberately narrow: an exit that reports *nothing* and writes nothing is also a contract
        // violation, but it is a different one, and the scope cannot tell it apart from a legitimate
        // silent success without knowing the exit code. Inventing a cause there would put a fabricated
        // error on a zero-exit path — measured on three commands that exit this way today — so those
        // are left to be fixed where they occur rather than papered over here.
    }

    /// <summary>
    /// Passes every write through to the real stdout and records that one happened. Only the fact is
    /// recorded, never the content — the scope decides whether to add a document, never what the
    /// command's own document should say.
    /// </summary>
    private sealed class ObservingWriter(TextWriter inner) : TextWriter
    {
        public bool WroteAnything { get; private set; }

        public override System.Text.Encoding Encoding => inner.Encoding;

        public override void Write(char value)
        {
            WroteAnything = true;
            inner.Write(value);
        }

        public override void Write(string? value)
        {
            if (value is null) return;
            WroteAnything = true;
            inner.Write(value);
        }

        public override void WriteLine(string? value)
        {
            WroteAnything = true;
            inner.WriteLine(value);
        }

        public override void Flush() => inner.Flush();
    }
}
