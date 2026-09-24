namespace MLoop.Core.Prediction;

/// <summary>
/// How a NER prediction — one tag per word — is written: the tags joined by spaces, the same shape
/// the label column is read in at training, so a prediction file can be compared with a training
/// file or fed back as one.
/// </summary>
/// <remarks>
/// ML.NET's NER trainer answers a word outside every entity with a missing key, which comes back as
/// an empty string. Joined as-is, "O B-DATE O" was written <c>"",B-DATE,""</c> and scored as wrong
/// against the <c>O</c> in the data. The outside tag is <see cref="Outside"/>, the IOB convention.
/// </remarks>
public static class TagSequence
{
    /// <summary>The tag of a word outside every entity.</summary>
    public const string Outside = "O";

    /// <summary>A predicted tag as it is written: a missing one is <see cref="Outside"/>.</summary>
    public static string Tag(string predicted) => string.IsNullOrEmpty(predicted) ? Outside : predicted;

    /// <summary>The tags of one sentence as one cell.</summary>
    public static string Render(IEnumerable<string> tags) => string.Join(' ', tags.Select(Tag));
}
