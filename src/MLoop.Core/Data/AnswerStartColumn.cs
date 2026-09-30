using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Core.Data;

/// <summary>
/// Finds the column that holds where a question-answering row's answer starts in its passage. The
/// trainer needs it and names it nothing in particular, so it is found by what it does rather than by
/// its name: on at least nine in ten sampled rows, the passage read from that position is the answer.
/// </summary>
public static class AnswerStartColumn
{
    private const int SampleRows = 200;
    private const double Required = 0.9;

    /// <summary>How far past the given position whitespace may be skipped — see <see cref="Locate"/>.</summary>
    private const int MaxWhitespaceSkip = 8;

    /// <summary>
    /// The first numeric column, other than those in <paramref name="exclude"/>, whose values locate
    /// <paramref name="answerColumn"/> inside <paramref name="contextColumn"/>; <c>null</c> when none does.
    /// </summary>
    public static string? Find(IDataView data, string contextColumn, string answerColumn, IReadOnlyCollection<string> exclude)
    {
        var contextCol = data.Schema[contextColumn];
        var answerCol = data.Schema[answerColumn];

        foreach (var candidate in data.Schema.Where(c => !c.IsHidden && !exclude.Contains(c.Name) && c.Type is NumberDataViewType))
        {
            using var cursor = data.GetRowCursor([contextCol, answerCol, candidate]);
            var contextGetter = cursor.GetGetter<ReadOnlyMemory<char>>(contextCol);
            var answerGetter = cursor.GetGetter<ReadOnlyMemory<char>>(answerCol);
            var position = Reader(cursor, candidate);

            ReadOnlyMemory<char> context = default, answer = default;
            int rows = 0, located = 0;
            while (rows < SampleRows && cursor.MoveNext())
            {
                contextGetter(ref context);
                answerGetter(ref answer);
                rows++;
                if (Locate(context.Span, answer.Span, position()) is not null)
                    located++;
            }

            if (rows > 0 && located >= Required * rows)
                return candidate.Name;
        }

        return null;
    }

    /// <summary>
    /// Where <paramref name="answer"/> starts in <paramref name="context"/>, given the recorded
    /// position <paramref name="at"/> — or <c>null</c> when the answer is not there.
    /// </summary>
    /// <remarks>
    /// Whitespace just past the recorded position is skipped. Data files are read with values trimmed,
    /// so an answer recorded as <c>" 서울"</c> arrives as <c>"서울"</c> while its position still points at the
    /// space — measured: 14 of 80 answers in a public Korean reading-comprehension set. The trimmed
    /// answer starts one character later, and that is the position a model has to learn.
    /// </remarks>
    public static int? Locate(ReadOnlySpan<char> context, ReadOnlySpan<char> answer, double at)
    {
        if (answer.IsEmpty || double.IsNaN(at) || at < 0 || at != Math.Floor(at) || at >= context.Length)
            return null;

        var i = (int)at;
        for (var skipped = 0; ; skipped++)
        {
            if (context[i..].StartsWith(answer))
                return i;
            if (skipped == MaxWhitespaceSkip || i + 1 >= context.Length || !char.IsWhiteSpace(context[i]))
                return null;
            i++;
        }
    }

    /// <summary>A reader for a numeric column as <see cref="double"/>, whatever its storage type.</summary>
    public static Func<double> Reader(DataViewRowCursor cursor, DataViewSchema.Column column)
    {
        var raw = column.Type.RawType;
        if (raw == typeof(float)) { var g = cursor.GetGetter<float>(column); float v = 0; return () => { g(ref v); return v; }; }
        if (raw == typeof(double)) { var g = cursor.GetGetter<double>(column); double v = 0; return () => { g(ref v); return v; }; }
        if (raw == typeof(int)) { var g = cursor.GetGetter<int>(column); int v = 0; return () => { g(ref v); return v; }; }
        if (raw == typeof(long)) { var g = cursor.GetGetter<long>(column); long v = 0; return () => { g(ref v); return v; }; }
        return () => -1;
    }
}
