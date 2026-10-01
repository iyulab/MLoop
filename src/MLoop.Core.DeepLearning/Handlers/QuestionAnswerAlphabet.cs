namespace MLoop.Core.DeepLearning;

/// <summary>
/// What ML.NET's question-answering model can write as an answer.
/// </summary>
/// <remarks>
/// The trainer does not cut its answer out of the passage; it decodes the answer's tokens with the
/// English RoBERTa tokenizer (<c>QATrainer</c>), created with unsupported characters filtered
/// (<c>EnglishRobertaTokenizer</c>'s default). That decoder turns each token character into the byte
/// it stands for — one character per byte — so nothing above U+00FF can come out. Measured on a Korean
/// reading-comprehension set: every predicted answer kept its digits and punctuation and lost all of
/// its Hangul. An answer outside Latin-1 cannot be matched exactly, whatever the model learned.
/// </remarks>
public static class QuestionAnswerAlphabet
{
    /// <summary>The highest character the decoder writes: one byte.</summary>
    public const char Highest = '\u00FF';

    /// <summary>Whether every character of <paramref name="answer"/> is one the model can write.</summary>
    public static bool CanWrite(ReadOnlySpan<char> answer)
    {
        foreach (var c in answer)
            if (c > Highest)
                return false;
        return true;
    }

    /// <summary>
    /// What to tell the user when some answers hold characters the model cannot write; <c>null</c> when
    /// it can write them all.
    /// </summary>
    public static string? Warning(int writable, int total)
    {
        if (total == 0 || writable == total)
            return null;

        var unwritable = total - writable;
        return $"[Warning] Question answering: {unwritable} of {total} answer(s) contain characters this model cannot write. "
            + "ML.NET's question-answering model writes its answer one byte per character (Latin-1), so Korean, Chinese, "
            + "Japanese, Cyrillic and similar text comes out with those characters missing. "
            + $"exact_match cannot exceed {(double)writable / total:P0} on this data, and char_f1 counts only what it can write.";
    }
}
