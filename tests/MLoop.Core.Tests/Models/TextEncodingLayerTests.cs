using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Embedding;
using MLoop.Core.Models;

namespace MLoop.Core.Tests.Models;

/// <summary>
/// The step in front of a model that reads encoded text: each encoded column gains its sentence vector beside it, the
/// text left as it was, and two columns of one encoder gain the cosine of their vectors — the same values whether a
/// whole view is encoded (batched) or one row is scored alone.
/// </summary>
public sealed class TextEncodingLayerTests : IDisposable
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures", "embedding");
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"mloop-layer-{Guid.NewGuid():N}");
    private readonly EmbedderPool _pool;

    public TextEncodingLayerTests()
    {
        var store = new EncoderStore(_root);
        store.Add(new EncoderDefinition("tiny-text", "1", 4, null, "input_ids", "last_hidden_state", EncoderPooling.MeanOfTokens,
            Kind: EncoderKind.Text, Text: new TextPreprocessing(12)),
            Path.Combine(Fixtures, "tiny-text-encoder.onnx"), Path.Combine(Fixtures, "tiny-tokenizer.model"));
        _pool = new EmbedderPool(store);
    }

    public void Dispose()
    {
        _pool.Dispose();
        if (Directory.Exists(_root))
            Directory.Delete(_root, recursive: true);
    }

    private sealed class Pair
    {
        public string First { get; set; } = "";

        public string Second { get; set; } = "";
    }

    private TextEncodingLayer Layer(MLContext ml)
    {
        var embedder = _pool.GetText("tiny-text").Embedder!;
        return new TextEncodingLayer(ml, [("First", embedder), ("Second", embedder)]);
    }

    private static readonly Pair[] Rows =
    [
        new() { First = "배송이 빨라서 좋았어요", Second = "배송이 빨라서 좋았어요" },
        new() { First = "the product is good", Second = "포장이 꼼꼼해요" },
    ];

    [Fact]
    public void EachColumnGainsItsVectorBesideTheText_AndAPairOfOneEncoderItsCosine()
    {
        var ml = new MLContext(1);
        var output = Layer(ml).Transform(ml.Data.LoadFromEnumerable(Rows));

        Assert.Equal(["배송이 빨라서 좋았어요", "the product is good"], output.GetColumn<string>("First"));   // the text stays
        var first = output.GetColumn<float[]>(TextEncodings.VectorColumnOf("First")).ToArray();
        var second = output.GetColumn<float[]>(TextEncodings.VectorColumnOf("Second")).ToArray();
        var cosine = output.GetColumn<float>(TextEncodings.CosineColumnOf("First", "Second")).ToArray();

        Assert.Equal(1f, cosine[0], 1e-5f);   // the same text, the same vector
        Assert.Equal(first[1].Zip(second[1], (a, b) => a * b).Sum(), cosine[1], 1e-5f);
        Assert.True(cosine[1] < 0.999f);
    }

    [Fact]
    public void ARowScoredAlone_GetsTheValuesTheBatchedViewGives()
    {
        var ml = new MLContext(1);
        var layer = Layer(ml);
        var batched = layer.Transform(ml.Data.LoadFromEnumerable(Rows)).GetColumn<float>(TextEncodings.CosineColumnOf("First", "Second")).ToArray();

        // A fresh layer has remembered nothing: the row mapper embeds the row it is given.
        var alone = Layer(ml);
        var view = ml.Data.LoadFromEnumerable(new[] { Rows[1] });
        var mapper = alone.GetRowToRowMapper(view.Schema);
        Assert.True(alone.IsRowToRowMapper);
        using var cursor = view.GetRowCursor(view.Schema);
        using var row = mapper.GetRow(cursor, mapper.OutputSchema);
        var getter = row.GetGetter<float>(mapper.OutputSchema[TextEncodings.CosineColumnOf("First", "Second")]);
        Assert.True(cursor.MoveNext());
        float value = 0;
        getter(ref value);

        Assert.Equal(batched[1], value, 1e-5f);
    }

    [Fact]
    public void ALayerIsNotSavedWithAModel()
    {
        var ml = new MLContext(1);
        var view = ml.Data.LoadFromEnumerable(Rows);

        Assert.Throws<NotSupportedException>(() => ml.Model.Save(Layer(ml), view.Schema, new MemoryStream()));
    }
}
