namespace MLoop.Core.DeepLearning;

/// <summary>
/// The one way a TorchSharp trainer is fitted: libtorch's random generator is put in a known state first, then the fit runs under
/// <see cref="PretrainedWeights.Guard{T}"/>.
/// </summary>
/// <remarks>
/// The same images, settings and <c>MLContext</c> seed gave a different detector in every process (mAP 0, 0.19, 0.31, 0.33 on
/// one 20-image set) while two fits in one process agreed: the trainer seeds libtorch itself, but something it draws before that
/// is not under that seed, and only the first fit of a process sees it. Seeding libtorch before the fit makes that first fit
/// repeat — four processes, one value. The seed does not make a fit good, only repeatable: a seed that trains a
/// detector which finds nothing finds nothing every time, so what a user does about it (train longer, change the data) can be
/// told apart from what luck did.
/// </remarks>
public static class TorchFit
{
    /// <summary>The seed libtorch is put in before every fit — the one every MLoop seed uses.</summary>
    public const int Seed = 42;

    /// <summary>Seeds libtorch, then runs <paramref name="fit"/> under the pretrained-weights guard.</summary>
    public static T Run<T>(Func<T> fit)
    {
        TorchSharp.torch.random.manual_seed(Seed);
        return PretrainedWeights.Guard(fit);
    }
}
