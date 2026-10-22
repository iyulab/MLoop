using System.Reflection;
using Microsoft.ML;

namespace MLoop.Search.Validation;

/// <summary>
/// Stops a fit in progress when a token is cancelled — the only way a search keeps its budget against a
/// trainer whose single fit is long.
/// </summary>
/// <remarks>
/// <para>
/// A token checked between folds cannot stop a fit already running, and one fit can be most of a budget:
/// SDCA sets its own pass limit (30,000 passes) when none is given, and on a small, nearly separable
/// partition it runs to that limit — measured at 42 s for one fit of fifty rows, so a five-fold trial took
/// over three minutes of a twenty-second budget. Capping the passes instead changes the model wherever
/// the solver needs them (a hundred passes left the same fit at AUC 0.5), which is a quality decision
/// the budget has no business making.
/// </para>
/// <para>
/// ML.NET checks for cancellation inside its training loops, but the switch, <c>MLContext.CancelExecution</c>,
/// is not public; ML.NET's own AutoML reached it as a friend assembly. It is called here by reflection, and
/// <see cref="IsSupported"/> says whether it was found — a test pins that, so an ML.NET upgrade that moves
/// it fails loudly instead of quietly falling back to stopping between folds.
/// </para>
/// <para>
/// Cancellation is permanent for the context: register only a context the caller is willing to discard.
/// </para>
/// </remarks>
internal static class FitCancellation
{
    private static readonly MethodInfo? CancelExecution = typeof(MLContext).GetMethod(
        "CancelExecution", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic, Type.EmptyTypes);

    /// <summary>Whether this ML.NET exposes the switch — without it, fits stop only between folds.</summary>
    internal static bool IsSupported => CancelExecution is not null;

    /// <summary>Cancels every fit running in <paramref name="mlContext"/> once <paramref name="cancellationToken"/> is cancelled.</summary>
    internal static CancellationTokenRegistration Register(MLContext mlContext, CancellationToken cancellationToken) =>
        CancelExecution is { } cancel && cancellationToken.CanBeCanceled
            ? cancellationToken.Register(() => cancel.Invoke(mlContext, null))
            : default;
}
