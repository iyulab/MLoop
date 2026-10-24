using Microsoft.ML;
using MLoop.Core.Diagnostics;

namespace MLoop.Core.DeepLearning;

/// <summary>
/// What ML.NET's pretrained text models can read and write: the characters their English RoBERTa tokenizer keeps.
/// </summary>
/// <remarks>
/// Text classification, sentence similarity, NER and question answering read their text through one tokenizer
/// (TorchSharp's <c>TokenizerExtensions</c>, an <c>EnglishRobertaTokenizer</c> created with unsupported characters
/// filtered). It maps each character below U+0100 to a byte and <b>drops every character above it</b> — on the way
/// in, not only out: «음악을 들으면서» becomes one space token. Measured on Korean news, dialogue and sentence sets:
/// 82–89% of the non-space characters, and 73–81% of the words, reach the model as nothing; digits, Latin letters
/// and punctuation are what is left. Question answering also writes its answer by decoding tokens, so an answer
/// outside Latin-1 cannot come out whole. <c>PretrainedTextAlphabetTests</c> checks the rule against the tokenizer.
/// </remarks>
public static class PretrainedTextAlphabet
{
    /// <summary>The highest character the tokenizer keeps: one byte.</summary>
    public const char Highest = 'ÿ';

    /// <summary>
    /// The share of a column's characters above which training warns. English text loses a little — curly quotes,
    /// dashes and emoji are above U+00FF too — and that is not what the warning is for.
    /// </summary>
    public const double WarnAbove = 0.05;

    /// <summary>Whether the model can read and write <paramref name="c"/>.</summary>
    public static bool Keeps(char c) => c <= Highest;

    /// <summary>
    /// What reads such text instead, for classification: the tabular classifiers featurize a text column by word and
    /// character n-grams over every character. On a Korean four-class sentence set the tabular multiclass model
    /// scored a macro accuracy of 0.72 on the same held-out rows where text classification scored 0.25 — chance.
    /// </summary>
    public const string ClassifyInstead =
        "A tabular classifier reads every character — train with --task multiclass-classification "
        + "(binary-classification for two classes): it featurizes a text column by word and character n-grams.";

    /// <summary>Whether every character of <paramref name="text"/> is one the model can write.</summary>
    public static bool CanWrite(ReadOnlySpan<char> text)
    {
        foreach (var c in text)
            if (!Keeps(c))
                return false;
        return true;
    }

    /// <summary>
    /// The share of <paramref name="column"/>'s non-space characters, over every row of <paramref name="data"/>,
    /// that the tokenizer drops; 0 for a column without one.
    /// </summary>
    public static double DroppedShare(IDataView data, string column)
    {
        var col = data.Schema[column];
        using var cursor = data.GetRowCursor([col]);
        var getter = cursor.GetGetter<ReadOnlyMemory<char>>(col);
        ReadOnlyMemory<char> value = default;
        long seen = 0, dropped = 0;
        while (cursor.MoveNext())
        {
            getter(ref value);
            foreach (var c in value.Span)
            {
                if (char.IsWhiteSpace(c))
                    continue;
                seen++;
                if (!Keeps(c))
                    dropped++;
            }
        }
        return seen == 0 ? 0 : (double)dropped / seen;
    }

    /// <summary>
    /// What to tell the user when the text <paramref name="task"/> trains on loses more than <see cref="WarnAbove"/>
    /// of <paramref name="column"/>'s characters; <c>null</c> otherwise. <paramref name="instead"/> names a task that
    /// reads the text, when there is one.
    /// </summary>
    public static string? ReadingWarning(string task, string column, double droppedShare, string? instead = null) =>
        droppedShare <= WarnAbove
            ? null
            : CoreMessage.Warning(
                $"{task}: {droppedShare:P0} of the characters in '{column}' are ones the model cannot read. "
                + "ML.NET's text models read text through an English tokenizer that drops every character above U+00FF — "
                + "Korean, Chinese, Japanese, Cyrillic and similar text reaches the model as nothing, and it learns from "
                + "what is left (digits, Latin letters, punctuation). Expect a weak model on this text."
                + (instead is null ? "" : " " + instead));

    /// <summary>
    /// What to tell the user when some answers hold characters the question-answering model cannot write; <c>null</c>
    /// when it can write them all.
    /// </summary>
    public static string? AnswerWarning(int writable, int total)
    {
        if (total == 0 || writable == total)
            return null;

        var unwritable = total - writable;
        return CoreMessage.Warning(
            $"Question answering: {unwritable} of {total} answer(s) contain characters this model cannot write. "
            + "ML.NET's question-answering model writes its answer one byte per character (Latin-1), so Korean, Chinese, "
            + "Japanese, Cyrillic and similar text comes out with those characters missing. "
            + $"exact_match cannot exceed {(double)writable / total:P0} on this data, and char_f1 counts only what it can write.");
    }
}
