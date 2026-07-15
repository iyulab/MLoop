using System.Runtime.CompilerServices;
using MLoop.Core.AutoML;

namespace MLoop.Tests;

/// <summary>
/// Registers the deep-learning module once when this test assembly loads, mirroring what
/// MLoop.CLI's Program.cs does at process startup (upstream-007 stage 2, task 5). MLoop.Core no
/// longer references TorchSharp/Vision directly (task 3) — DL task dispatch (AutoMLRunner's
/// switch, DataLoaderFactory's directory-based loaders) goes through <see cref="DeepLearningRegistry"/>,
/// which stays null until something registers a module. Without this, every DL-task test here
/// (image-classification / object-detection round trips through TrainingEngine) would fail before
/// reaching the runtime-install gate they're actually exercising.
/// </summary>
internal static class TestModuleInitializer
{
    [ModuleInitializer]
    public static void Initialize()
    {
        DeepLearningRegistry.Register(new MLoop.Core.DeepLearning.DeepLearningModule());
    }
}
