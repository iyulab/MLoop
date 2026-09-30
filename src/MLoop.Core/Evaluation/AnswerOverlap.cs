using System.Globalization;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Core.Evaluation;

/// <summary>
/// Scores a question-answering model by how much its answer overlaps the expected one. Training and
/// <c>mloop evaluate</c> both score with this, so the number a model was promoted on is the number
/// evaluation reports.
/// </summary>
/// <remarks>
/// The rules are those of the KorQuAD v1 evaluation script: answers are normalized (lowercased,
/// quotes and brackets and punctuation removed, whitespace collapsed); <em>exact match</em> is equality
/// after normalization; <em>F1</em> is computed over characters, not words. The character form was
/// adopted for Korean, where an answer and a word rarely coincide — `서울에서` holds the answer `서울` — and a
/// word-level F1 scores such an answer zero. It also reads languages without spaces, which is why it is
/// the rule here rather than a per-language choice.
/// </remarks>
public static class AnswerOverlap
{
    /// <summary>Metric names as they are recorded.</summary>
    public const string ExactMatch = "exact_match";
    public const string CharF1 = "char_f1";

    /// <summary>The column the question-answering trainer writes its answers to, best first.</summary>
    public const string PredictedAnswerColumn = "Answer";

    /// <summary>
    /// Mean exact match and character F1 over the rows of <paramref name="scored"/> — a model's output
    /// over data that still carries the expected answer in <paramref name="expectedColumn"/>. Only the
    /// model's best answer counts.
    /// </summary>
    public static (double ExactMatch, double F1) Measure(
        IDataView scored, string expectedColumn, string predictedColumn = PredictedAnswerColumn)
    {
        var expectedCol = scored.Schema[expectedColumn];
        var predictedCol = scored.Schema[predictedColumn];
        using var cursor = scored.GetRowCursor([expectedCol, predictedCol]);
        var expectedGetter = cursor.GetGetter<ReadOnlyMemory<char>>(expectedCol);
        var predictedGetter = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(predictedCol);

        ReadOnlyMemory<char> expected = default;
        VBuffer<ReadOnlyMemory<char>> predicted = default;
        double exact = 0, f1 = 0;
        var rows = 0;
        while (cursor.MoveNext())
        {
            expectedGetter(ref expected);
            predictedGetter(ref predicted);
            var best = predicted.Length > 0 ? predicted.GetItemOrDefault(0).ToString() : string.Empty;
            var truth = expected.ToString();
            exact += IsExactMatch(best, truth) ? 1 : 0;
            f1 += CharacterF1(best, truth);
            rows++;
        }

        return rows == 0 ? (0, 0) : (exact / rows, f1 / rows);
    }

    /// <summary>Whether two answers are equal after normalization.</summary>
    public static bool IsExactMatch(string prediction, string truth) =>
        Normalize(prediction) == Normalize(truth);

    /// <summary>
    /// F1 over the characters of the two normalized answers, spaces excluded: the share of characters
    /// they have in common, balanced between the prediction's and the truth's length.
    /// </summary>
    public static double CharacterF1(string prediction, string truth)
    {
        var p = Normalize(prediction).Replace(" ", "", StringComparison.Ordinal);
        var t = Normalize(truth).Replace(" ", "", StringComparison.Ordinal);
        if (p.Length == 0 || t.Length == 0)
            return 0;

        var counts = new Dictionary<char, int>();
        foreach (var c in t)
            counts[c] = counts.GetValueOrDefault(c) + 1;

        var common = 0;
        foreach (var c in p)
        {
            if (counts.TryGetValue(c, out var n) && n > 0)
            {
                counts[c] = n - 1;
                common++;
            }
        }

        if (common == 0)
            return 0;
        var precision = (double)common / p.Length;
        var recall = (double)common / t.Length;
        return 2 * precision * recall / (precision + recall);
    }

    /// <summary>
    /// The KorQuAD v1 normalization: quotes and CJK brackets become spaces, the text is lowercased,
    /// punctuation is removed, and runs of whitespace become one space.
    /// </summary>
    public static string Normalize(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (var raw in text.ToLowerInvariant())
        {
            var c = raw;
            if (c is '\'' or '"' or '《' or '》' or '<' or '>' or '〈' or '〉' or '(' or ')' or '‘' or '’' or '“' or '”')
                c = ' ';
            else if (char.IsPunctuation(c) || CharUnicodeInfo.GetUnicodeCategory(c) == UnicodeCategory.MathSymbol
                     || c is '$' or '^' or '`' or '|' or '~')
                continue;
            sb.Append(char.IsWhiteSpace(c) ? ' ' : c);
        }

        return string.Join(' ', sb.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries));
    }
}
