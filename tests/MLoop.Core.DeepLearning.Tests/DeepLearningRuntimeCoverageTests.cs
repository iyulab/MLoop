using MLoop.Core.DeepLearning;
using MLoop.Core.Models;
using MLoop.Core.Runtime;

namespace MLoop.Core.DeepLearning.Tests;

/// <summary>
/// Every task the deep-learning module trains needs a native runtime, and every native runtime
/// declares which tasks require it. Those two statements live in different assemblies and nothing
/// but this test connects them.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="DeepLearningModule"/> decides whether it can handle a task;
/// <see cref="RuntimeRegistry"/> decides which runtime a task needs to have installed first. Add a
/// seventh deep-learning task to the module and forget the registry entry, and the failure is not a
/// build break — it is a user who runs <c>mloop train --task …</c>, is never told a runtime is
/// missing, and gets the native load error instead of the install prompt the CLI exists to give.
/// The reverse omission is quieter still: a runtime claiming a task the module will refuse.
/// </para>
/// <para>
/// Stated as a relationship rather than converged into one list, deliberately. The registry knows
/// more than "is this deep learning" — it knows <i>which</i> runtime, per platform — so collapsing
/// the two into a shared set would throw away the half that matters and leave the same drift one
/// layer down. What has to hold is that the two agree on their common axis.
/// </para>
/// </remarks>
public class DeepLearningRuntimeCoverageTests
{
    private static readonly DeepLearningModule Module = new();

    private static string[] TasksRequiringARuntime =>
        [.. RuntimeRegistry.All.SelectMany(r => r.RequiredByTasks).Distinct()];

    [Fact]
    public void EveryTaskARuntimeClaimsIsOneTheModuleTrains()
    {
        var claimedButUnhandled = TasksRequiringARuntime
            .Where(t => !Module.CanHandleTask(t))
            .ToArray();

        Assert.True(
            claimedButUnhandled.Length == 0,
            "A runtime declares RequiredByTasks entries the deep-learning module will not train: "
            + string.Join(", ", claimedButUnhandled));
    }

    [Fact]
    public void EveryTaskTheModuleTrainsHasARuntimeThatServesIt()
    {
        var handledWithoutRuntime = TaskTypes.All
            .Where(t => Module.CanHandleTask(t) && RuntimeRegistry.GetRequiredByTask(t) is null)
            .ToArray();

        Assert.True(
            handledWithoutRuntime.Length == 0,
            "The deep-learning module trains tasks no runtime declares, so the CLI cannot tell a "
            + "user what to install: " + string.Join(", ", handledWithoutRuntime));
    }

    /// <summary>
    /// A deep-learning task is still a task. The module answering "yes" for a value the vocabulary
    /// does not admit would mean a task reachable by the trainer that <c>validate</c> rejects.
    /// </summary>
    [Fact]
    public void TheModuleOnlyClaimsTasksTheVocabularyAdmits()
    {
        var unknown = TasksRequiringARuntime.Where(t => !TaskTypes.IsValid(t)).ToArray();

        Assert.True(
            unknown.Length == 0,
            "Runtime definitions name tasks absent from MLoop.Core.Models.TaskTypes: "
            + string.Join(", ", unknown));
    }
}
