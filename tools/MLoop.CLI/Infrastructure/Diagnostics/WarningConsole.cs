using Spectre.Console;

namespace MLoop.CLI.Infrastructure.Diagnostics;

/// <summary>
/// The single seam a training warning goes through, mirroring <see cref="ErrorConsole"/> for the
/// non-fatal channel. A warning renders as a <c>Warning:</c> line for a human and, in machine mode,
/// is reported as a <c>warning</c> event — from one call site, so the two audiences cannot drift.
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
    /// <summary>
    /// Renders a <c>Warning:</c> line (ambient console — silenced in machine mode) and reports the
    /// same text as a <c>warning</c> event when a machine-output scope is active.
    /// <paramref name="markup"/> is Spectre markup — escape interpolated user data with
    /// <see cref="Markup.Escape"/>.
    /// </summary>
    public static void Warn(string markup)
    {
        AnsiConsole.MarkupLine($"[yellow]Warning:[/] {markup}");
        MachineOutputScope.ReportWarning(markup);
    }
}
