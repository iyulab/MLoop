namespace MLoop.Core.Embedding;

/// <summary>
/// The encoders <c>mloop encoder install</c> knows by name. Each entry pins where its weights are downloaded from and
/// their SHA-256, so an install is the same file on every machine — the product ships no weights.
/// </summary>
public static class EncoderCatalog
{
    /// <summary>
    /// DINOv2 ViT-S/14 (<c>facebook/dinov2-small</c>, Apache-2.0) as ONNX: 384 dimensions, the class token of
    /// <c>last_hidden_state</c>, preprocessed as its image processor does (shortest edge 256, bicubic; centre crop
    /// 224; ImageNet mean and std); its 256 patch tokens are a 16 × 16 grid of 14-pixel patches. The download is pinned
    /// to one revision of <c>onnx-community/dinov2-small</c>.
    /// </summary>
    public static readonly EncoderDefinition DinoV2Small = new(
        Id: "dinov2-small",
        Version: "1",
        Dimension: 384,
        Preprocessing: ImagePreprocessing.ImageNet(shortestEdge: 256, cropSize: 224),
        InputName: "pixel_values",
        OutputName: "last_hidden_state",
        Pooling: EncoderPooling.FirstToken,
        Url: "https://huggingface.co/onnx-community/dinov2-small/resolve/8b1f705a3a7f6f062f6bdd21986c1583d3ef105d/onnx/model.onnx",
        Sha256: "f22797eabf810a75e41de68d378541ebea372122b25c4ce3ef25ff618250c20a",
        PatchSize: 14);

    /// <summary>
    /// paraphrase-multilingual-MiniLM-L12-v2 (<c>sentence-transformers</c>, Apache-2.0) as ONNX: a sentence encoder of
    /// some fifty languages, Korean among them — 384 dimensions, the mean of <c>last_hidden_state</c> over the text's
    /// tokens, at most 128 tokens (its <c>max_seq_length</c>), XLM-RoBERTa's SentencePiece vocabulary. The downloads are
    /// pinned to one revision of the model's own repository.
    /// </summary>
    public static readonly EncoderDefinition MultilingualMiniLm = new(
        Id: "multilingual-minilm",
        Version: "1",
        Dimension: 384,
        Preprocessing: null,
        InputName: "input_ids",
        OutputName: "last_hidden_state",
        Pooling: EncoderPooling.MeanOfTokens,
        Url: "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/e8f8c211226b894fcb81acc59f3b34ba3efd5f42/onnx/model.onnx",
        Sha256: "10f7a088420252b26caf819236ca2c9d2987afd0fc06fec7553b542a5655a05a",
        Kind: EncoderKind.Text,
        Text: new TextPreprocessing(MaxTokens: 128, TokenIds: TextTokenIds.XlmRoberta),
        TokenizerUrl: "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/e8f8c211226b894fcb81acc59f3b34ba3efd5f42/sentencepiece.bpe.model",
        TokenizerSha256: "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865");

    /// <summary>Every entry, by id.</summary>
    public static IReadOnlyList<EncoderDefinition> All { get; } = [DinoV2Small, MultilingualMiniLm];

    /// <summary>The entry named <paramref name="id"/>, or <see langword="null"/>.</summary>
    public static EncoderDefinition? Find(string? id) =>
        All.FirstOrDefault(e => string.Equals(e.Id, id?.Trim(), StringComparison.OrdinalIgnoreCase));

    /// <summary>The ids, for a message that lists what can be installed.</summary>
    public static string Ids => string.Join(", ", All.Select(e => e.Id));

    /// <summary>
    /// The command that installs a catalog encoder — of <paramref name="kind"/>, or of any kind — for a message to give
    /// as is: the id itself when there is one, the choices as a placeholder (<c>&lt;a|b&gt;</c>) when there are more;
    /// <see langword="null"/> when the catalog has none of the kind.
    /// </summary>
    public static string? InstallCommand(EncoderKind? kind = null)
    {
        var ids = All.Where(e => kind is null || e.Kind == kind).Select(e => e.Id).ToList();
        return ids.Count switch
        {
            0 => null,
            1 => $"mloop encoder install {ids[0]}",
            _ => $"mloop encoder install <{string.Join('|', ids)}>",
        };
    }
}
