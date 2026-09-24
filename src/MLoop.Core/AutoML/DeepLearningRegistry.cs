namespace MLoop.Core.AutoML;

/// <summary>
/// Process-wide registration point for the optional <see cref="IDeepLearningModule"/>.
/// MLoop.CLI / MLoop.API call <see cref="Register"/> once at startup. Multi-process casual
/// design: each process registers independently; no cross-process state.
/// </summary>
public static class DeepLearningRegistry
{
    private static IDeepLearningModule? _module;

    public static void Register(IDeepLearningModule module) => _module = module;
    public static IDeepLearningModule? Current => _module;
    public static bool IsRegistered => _module is not null;

    /// <summary>
    /// Whether <paramref name="task"/> is trained by the deep-learning module — a fit that runs a
    /// fixed number of epochs and takes no time budget, so nothing about the budget (an estimate, a
    /// probe, a limit) applies to it.
    /// </summary>
    public static bool Handles(string? task) =>
        !string.IsNullOrWhiteSpace(task) && _module?.CanHandleTask(task.Trim()) == true;
}
