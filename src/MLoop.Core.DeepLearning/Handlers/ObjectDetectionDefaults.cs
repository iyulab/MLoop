namespace MLoop.Core.DeepLearning;

/// <summary>What an object detector is trained with where MLoop chooses and the trainer's own default did not hold up.</summary>
public static class ObjectDetectionDefaults
{
    /// <summary>
    /// The SGD learning rate. The trainer's default is 1.0, and at 1.0 the loss fell for four epochs and then rose: on a noisy
    /// 60-image set it ended at 1.1–1.7 and the detector scored <c>map_50</c> 0 at the 0.5 score threshold in all three seeds tried
    /// (0, 0, 0.04), and 0 on a second such set. At 0.3 the loss fell every epoch to about 0.16 on those sets and the same three seeds
    /// scored <c>map_50</c> 0.96, 0.94 and 0.96 — but 0.3 is itself near the edge: on small sets with small, cluttered objects (32 images,
    /// 384 px, unannotated look-alikes) the loss turned up at the third to fifth epoch, before the trainer's step down at epoch 6, and
    /// ended at 2.1–2.5 with <c>map_50</c> 0; and on an easy set a run of 40 epochs with the step moved later diverged after the twelfth.
    /// At 0.15 the loss fell every epoch on all of them, for 10 and for 40 epochs (<c>map_50</c> 10 → 40 epochs: 0.42 → 0.58 on the
    /// cluttered set, 0.89 → 0.90 on a mid set, 1.0 → 1.0 on the easy one); 0.1 was stable but slower (0.08 on the cluttered set at 10
    /// epochs, 0.77 on the mid set), 0.03 learned too slowly (0.10). 0.15 is the best of the rates tried (1.0, 0.3, 0.15, 0.1, 0.03),
    /// not a searched optimum, and the sets were synthetic. The step down at epoch 6 stays where the trainer puts it: it is what
    /// keeps a longer run stable, so it is not scaled with the epoch count.
    /// </summary>
    public const double InitLearningRate = 0.15;
}
