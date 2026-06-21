using MLoop.Core.Runtime;

namespace MLoop.Core.Tests.Runtime;

/// <summary>
/// Tests for <see cref="RuntimeManager.EnsureRuntimeForTask"/> — the shared guard that loads a
/// task's native runtime before any model load deserializes native parameters (BUG-40). The guard
/// must be a no-op for tabular tasks (so it can be called unconditionally on every load path) and
/// must fail loudly with an install hint for an uninstalled DL runtime.
/// </summary>
public class RuntimeManagerTests
{
    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void EnsureRuntimeForTask_NullOrBlank_IsNoOp(string? taskType)
    {
        // Must never throw — inference paths call this even when the task is unknown/unset.
        RuntimeManager.EnsureRuntimeForTask(taskType!);
    }

    [Theory]
    [InlineData("regression")]
    [InlineData("binary-classification")]
    [InlineData("multiclass-classification")]
    [InlineData("anomaly-detection")]
    [InlineData("clustering")]
    [InlineData("forecasting")]
    [InlineData("not-a-real-task")]
    public void EnsureRuntimeForTask_TaskWithoutNativeRuntime_IsNoOp(string taskType)
    {
        // Tabular/unknown tasks require no on-demand runtime, so the guard must do nothing —
        // this is what makes it safe to call unconditionally before every Model.Load.
        RuntimeManager.EnsureRuntimeForTask(taskType);
    }

    [Theory]
    [InlineData("image-classification")]
    [InlineData("object-detection")]
    public void EnsureRuntimeForTask_DlTask_LoadsOrThrowsInstallHint(string taskType)
    {
        var runtime = RuntimeRegistry.GetRequiredByTask(taskType);
        Assert.NotNull(runtime);

        if (new RuntimeManager().IsInstalled(runtime!))
        {
            // Installed: the guard registers the native search paths and returns without throwing.
            RuntimeManager.EnsureRuntimeForTask(taskType);
        }
        else
        {
            // Not installed: the guard must fail loudly with an actionable install hint rather than
            // letting the load reach a cryptic DllNotFoundException at deserialization time.
            var ex = Assert.Throws<InvalidOperationException>(
                () => RuntimeManager.EnsureRuntimeForTask(taskType));

            Assert.Contains("runtime", ex.Message, StringComparison.OrdinalIgnoreCase);
            Assert.Contains($"mloop runtime install {runtime!.Id}", ex.Message, StringComparison.OrdinalIgnoreCase);
        }
    }
}
