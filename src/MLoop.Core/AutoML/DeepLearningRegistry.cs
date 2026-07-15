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
}
