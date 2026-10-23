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
    /// 224; ImageNet mean and std). The download is pinned to one revision of <c>onnx-community/dinov2-small</c>.
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
        Sha256: "f22797eabf810a75e41de68d378541ebea372122b25c4ce3ef25ff618250c20a");

    /// <summary>Every entry, by id.</summary>
    public static IReadOnlyList<EncoderDefinition> All { get; } = [DinoV2Small];

    /// <summary>The entry named <paramref name="id"/>, or <see langword="null"/>.</summary>
    public static EncoderDefinition? Find(string? id) =>
        All.FirstOrDefault(e => string.Equals(e.Id, id?.Trim(), StringComparison.OrdinalIgnoreCase));

    /// <summary>The ids, for a message that lists what can be installed.</summary>
    public static string Ids => string.Join(", ", All.Select(e => e.Id));
}
