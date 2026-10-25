using Microsoft.ML.Tokenizers;

namespace MLoop.Core.Embedding;

/// <summary>
/// Turns text into a text encoder's token ids (<see cref="TextPreprocessing"/>): the tokenizer model normalises and cuts
/// it into pieces (SentencePiece, through <c>Microsoft.ML.Tokenizers</c>), the pieces are numbered as the encoder's
/// vocabulary numbers them, and the sentence markers go around them. Safe to share across threads.
/// </summary>
public sealed class TextPreprocessor
{
    private readonly SentencePieceTokenizer _tokenizer;
    private readonly TextPreprocessing _text;

    private TextPreprocessor(SentencePieceTokenizer tokenizer, TextPreprocessing text)
    {
        _tokenizer = tokenizer;
        _text = text;
    }

    /// <summary>
    /// How MLoop turns text into ids, beside the vocabulary's numbering: the same weights and tokenizer give other vectors
    /// when it changes, so it is part of the space's identity (<see cref="EncoderIdentity.Preprocessor"/>).
    /// </summary>
    public static string Revision(TextTokenIds ids) => ids switch
    {
        TextTokenIds.XlmRoberta => "xlm-roberta/1",
        _ => throw new ArgumentOutOfRangeException(nameof(ids), ids, null),
    };

    /// <summary>The id of the padding token, which fills a batch's shorter texts.</summary>
    public long PadId => _text.TokenIds switch
    {
        TextTokenIds.XlmRoberta => 1,
        _ => throw new InvalidOperationException($"Unknown token numbering {_text.TokenIds}."),
    };

    /// <summary>Reads the SentencePiece model at <paramref name="tokenizerPath"/> for <paramref name="text"/>.</summary>
    /// <exception cref="InvalidOperationException">The file is not a SentencePiece model.</exception>
    public static TextPreprocessor Open(string tokenizerPath, TextPreprocessing text)
    {
        ArgumentNullException.ThrowIfNull(text);
        try
        {
            using var stream = File.OpenRead(tokenizerPath);
            return new TextPreprocessor(
                SentencePieceTokenizer.Create(stream, addBeginningOfSentence: false, addEndOfSentence: false), text);
        }
        catch (Exception ex) when (ex is not FileNotFoundException and not DirectoryNotFoundException and not UnauthorizedAccessException)
        {
            throw new InvalidOperationException($"'{tokenizerPath}' is not a SentencePiece tokenizer model: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// The ids of <paramref name="text"/> — the prefix, then the text — between the sentence markers, at most
    /// <see cref="TextPreprocessing.MaxTokens"/> of them; <c>Truncated</c> says whether pieces were cut off the end.
    /// </summary>
    public (long[] Ids, bool Truncated) Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var pieces = UnknownRunsMerged(_tokenizer.EncodeToIds(_text.Prefix + text), _tokenizer.UnknownId);
        var room = _text.MaxTokens - 2;
        var kept = Math.Min(pieces.Count, room);
        var ids = new long[kept + 2];
        switch (_text.TokenIds)
        {
            case TextTokenIds.XlmRoberta:
                // fairseq: <s> 0 · <pad> 1 · </s> 2 · <unk> 3, every other piece one above its SentencePiece number
                // (SentencePiece itself has <unk> 0 · <s> 1 · </s> 2).
                ids[0] = 0;
                for (var i = 0; i < kept; i++)
                    ids[i + 1] = pieces[i] == _tokenizer.UnknownId ? 3 : pieces[i] + 1L;
                ids[^1] = 2;
                break;
            default:
                throw new InvalidOperationException($"Unknown token numbering {_text.TokenIds}.");
        }
        return (ids, pieces.Count > room);
    }

    // SentencePiece's Unigram encoder merges a run of characters it has no piece for into one unknown piece (the encoders'
    // tokenizers — sentencepiece, HF tokenizers — do); Microsoft.ML.Tokenizers 2.0.1 gives one per character, so
    // "魑魅" would be two <unk> where the encoder was trained on one.
    // TODO(upstream: https://github.com/dotnet/machinelearning/issues/7859) — drop once Microsoft.ML.Tokenizers merges a run of unknown pieces itself (TextEmbedderTests pins the ids).
    private static IReadOnlyList<int> UnknownRunsMerged(IReadOnlyList<int> pieces, int unknown)
    {
        var merged = new List<int>(pieces.Count);
        foreach (var piece in pieces)
        {
            if (piece != unknown || merged.Count == 0 || merged[^1] != unknown)
                merged.Add(piece);
        }
        return merged;
    }
}
