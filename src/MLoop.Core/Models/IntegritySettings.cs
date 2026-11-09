namespace MLoop.Core.Models;

/// <summary>
/// <c>integrity:</c> in mloop.yaml — what a load does when the production model is not the file its promotion recorded
/// (<see cref="ModelIntegrity"/>). Absent: the model answers and the mismatch is said, as before.
/// </summary>
public sealed class IntegritySettings
{
    /// <summary>
    /// <c>true</c>: a production model that changed after it was promoted is refused (<see cref="ChangedModelException"/>) instead of
    /// answering with a warning. Off by default — placing a model by hand is a way some projects work.
    /// </summary>
    public bool RefuseChangedModel { get; set; }
}
