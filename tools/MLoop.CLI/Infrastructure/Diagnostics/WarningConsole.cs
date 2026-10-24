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
    /// <summary>
    /// Renders a <c>Warning:</c> line (ambient console — silenced in machine mode) and reports the
    /// same text as a <c>warning</c> event when a machine-output scope is active.
    /// <paramref name="markup"/> is Spectre markup — escape interpolated user data with
    /// <see cref="Markup.Escape"/>.
    /// </summary>
    public static void Warn(string markup)
    {
        markup = WithoutOwnWarningSign(markup);
        AnsiConsole.MarkupLine($"[yellow]Warning:[/] {markup}");
        MachineOutputScope.ReportWarning(markup);
        JsonOutputScope.ReportWarning(markup);
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
