namespace MLoop.Core.DeepLearning;

/// <summary>What an object detector is trained with where MLoop chooses and the trainer's own default did not hold up.</summary>
public static class ObjectDetectionDefaults
{
    /// <summary>
    /// The SGD learning rate. The trainer's default is 1.0, and at 1.0 the loss fell for four epochs and then rose: on a noisy
    /// 60-image set it ended at 1.1–1.7 and the detector scored <c>map_50</c> 0 at the 0.5 score threshold in all three seeds tried
    /// (0, 0, 0.04), and 0 on a second such set. At 0.3 the loss fell every epoch to about 0.16 and the same three seeds scored
    /// <c>map_50</c> 0.96, 0.94 and 0.96 (0.92 on the second set); 0.1 scored 0.78–0.87 and 0.03 learned too slowly (0.10). On a
    /// 20-image easy set 1.0 and 0.3 were alike (0.375 at <c>map_50</c> both). 0.3 is the best of the four rates tried, not a searched
    /// optimum, and the sets were synthetic.
    /// </summary>
    public const double InitLearningRate = 0.3;
}
