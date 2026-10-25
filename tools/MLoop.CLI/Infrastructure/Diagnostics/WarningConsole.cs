using Spectre.Console;

namespace MLoop.CLI.Infrastructure.Diagnostics;

/// <summary>
/// The single seam a training warning goes through, mirroring <see cref="ErrorConsole"/> for the
/// non-fatal channel. A warning renders as a <c>Warning:</c> line for a human and, in machine mode,
/// is reported as a <c>warning</c> event (<c>train</c>'s stream) or kept for the command's document
/// (<see cref="JsonOutputScope.Warnings"/>) — from one call site, so the audiences cannot drift.
/// </summary>
/// <remarks>
/// <para>
/// Warnings deliberately stay on <b>stdout</b>, unlike errors. A successful run must leave stderr
/// empty — consumers read any stderr as a failure signal (the other half of the 0.28.0 channel
/// contract) — and a warning is by definition something the run recovered from.
/// </para>
/// <para>
/// Before this seam existed, warnings were raised from scattered <c>Console.WriteLine("[Warning] …")</c>
/// and Spectre <c>[yellow]Warning:[/]</c> calls across three assemblies. In machine mode all of those
/// are silenced with the rest of the narration, so a <c>--json</c> consumer saw <em>fewer</em>
/// warnings than a human at the terminal — and the downstream consumer had already declared they
/// will not parse <c>[Warning]</c> prose out of a captured stream.
/// </para>
/// </remarks>
public static class WarningConsole
{
    // The collection open in this async flow, if any — flows into Task.Run and awaits, and not into another
    // request's flow, which is what lets a server keep one job's warnings apart from the rest of its traffic.
    private static readonly AsyncLocal<WarningCollection?> CurrentCollection = new();

    /// <summary>
    /// Collects every warning raised from here on in this async flow — the work it awaits and the tasks it starts
    /// included — until the collection is disposed. <see cref="JsonOutputScope"/> keeps a command's warnings for its
    /// document, and is one per process as a CLI command is; a server running a training job beside other requests
    /// needs the job's own, and this is them.
    /// </summary>
    public static WarningCollection Collect()
    {
        var collection = new WarningCollection(CurrentCollection.Value);
        CurrentCollection.Value = collection;
        return collection;
    }

    /// <summary>The warnings raised while a <see cref="Collect"/> was open, plain text, in order.</summary>
    public sealed class WarningCollection : IDisposable
    {
        private readonly List<string> _warnings = [];
        private readonly WarningCollection? _outer;
        private bool _disposed;

        internal WarningCollection(WarningCollection? outer) => _outer = outer;

        public IReadOnlyList<string> Warnings
        {
            get
            {
                lock (_warnings)
                    return [.. _warnings];
            }
        }

        internal void Add(string warning)
        {
            lock (_warnings)
                _warnings.Add(warning);
            _outer?.Add(warning);
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            if (ReferenceEquals(CurrentCollection.Value, this))
                CurrentCollection.Value = _outer;
        }
    }

    /// <summary>
    /// <see cref="Warn(string)"/> from <c>$"…"</c>: the literal is markup, each interpolated value is text
    /// (<see cref="MarkupText"/>) — a path or name with brackets is shown as given.
    /// </summary>
    public static void Warn(MarkupText markup) => Warn(markup.ToString());

    /// <summary>
    /// Renders a <c>Warning:</c> line (ambient console — silenced in machine mode) and reports the
    /// same text as a <c>warning</c> event when a machine-output scope is active.
    /// <paramref name="markup"/> is Spectre markup, used as given — a string built elsewhere escapes its user data with
    /// <see cref="Markup.Escape"/>; <c>$"…"</c> goes to <see cref="Warn(MarkupText)"/>, which does.
    /// </summary>
    public static void Warn(string markup)
    {
        markup = WithoutOwnWarningSign(markup);
        AnsiConsole.MarkupLine($"[yellow]Warning:[/] {markup}");
        MachineOutputScope.ReportWarning(markup);
        JsonOutputScope.ReportWarning(markup);
        if (CurrentCollection.Value is { } collection)
            collection.Add(Plain(markup));
    }

    private static string Plain(string markup)
    {
        try { return Markup.Remove(markup); }
        catch (InvalidOperationException) { return markup; } // not valid markup — kept verbatim
    }

    /// <summary>
    /// The message without a warning sign of its own at the start (<c>⚠</c>, <c>⚠️</c>): this seam already
    /// says «Warning:», and the many messages written with a sign read <c>Warning: ⚠ …</c> — two layers
    /// adding the same decoration, which the release walk counts as a defect. Other leading symbols
    /// (a severity colour) say something the prefix does not and stay.
    /// </summary>
    internal static string WithoutOwnWarningSign(string message)
    {
        var trimmed = message.TrimStart();
        if (!trimmed.StartsWith('⚠'))
            return message;
        trimmed = trimmed[1..];
        if (trimmed.StartsWith('\uFE0F'))
            trimmed = trimmed[1..];
        return trimmed.TrimStart();
    }
}
