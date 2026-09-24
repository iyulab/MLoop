namespace MLoop.Core.Data;

/// <summary>
/// Whether a string column holds natural-language text rather than categorical codes — the one
/// judgement behind featurizing a column as text, labelling it "Text Feature" in <c>info</c>, and
/// choosing which columns a text model reads.
/// </summary>
public static class TextLikeness
{
    /// <summary>How many values the token and length criteria look at.</summary>
    public const int SampleSize = 200;

    /// <summary>
    /// Judges a column from its values.
    /// </summary>
    /// <param name="sampleValues">Non-empty values, in row order; only the first
    /// <see cref="SampleSize"/> are read.</param>
    /// <param name="uniqueCount">Distinct non-empty values among the <paramref name="rowCount"/> rows.</param>
    /// <param name="rowCount">Rows the unique count was taken over.</param>
    public static bool LooksLikeText(IReadOnlyList<string> sampleValues, int uniqueCount, int rowCount)
    {
        if (rowCount == 0)
            return false;

        // Most values distinct: free text, not a code list.
        double uniqueRatio = (double)uniqueCount / rowCount;
        if (uniqueRatio > 0.5)
            return true;

        int totalTokens = 0;
        int totalLength = 0;
        int validCount = 0;
        foreach (var value in sampleValues.Take(SampleSize))
        {
            validCount++;
            totalLength += value.Length;
            totalTokens += value.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
        }

        if (validCount == 0)
            return false;

        double avgTokens = (double)totalTokens / validCount;
        double avgLength = (double)totalLength / validCount;

        // Three or more words per value reads as language.
        if (avgTokens >= 3)
            return true;

        // Long strings are text even without spaces.
        if (avgLength > 30)
            return true;

        // Many distinct values at a moderate ratio.
        return uniqueCount > 200 && uniqueRatio > 0.1;
    }
}
