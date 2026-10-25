using System.Collections.Concurrent;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Embedding;
using MLoop.Core.Storage;

namespace MLoop.Core.Models;

/// <summary>
/// The text columns a model also reads through a text encoder (<see cref="EncoderKind.Text"/>): each is a text column
/// of the data, featurized as any text column is, and the model was trained on its sentence vector too — a column
/// beside it (<see cref="VectorColumnOf"/>). It lives beside the model
/// (<see cref="ExperimentLayout.TextEncodersFileName"/>) and <see cref="ScoringModel.Load(MLContext, string, string?, ScoringLayers)"/>
/// puts the same encoding in front of the model, so predict, serve and evaluate give the model the vectors it was
/// trained on from the text a row carries — and refuse when the installed encoder is not the one it was trained with.
/// </summary>
/// <param name="Columns">Each encoded column with the encoder that encoded it.</param>
public sealed record TextEncodings(IReadOnlyList<EncodedTextColumn> Columns)
{
    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    /// <summary>The tasks whose models read encoded columns: those the trainer search fits on feature columns.</summary>
    public static IReadOnlyList<string> Tasks { get; } = ["binary-classification", "multiclass-classification", "regression"];

    /// <summary>
    /// The column a text column's sentence vector is read from: <c>&lt;column&gt;.embedding</c>, added beside the text, which
    /// keeps its own featurization — the words and characters of a text carry what a sentence vector drops (its style,
    /// its endings), and the vector carries what they cannot (meaning across paraphrases).
    /// </summary>
    public static string VectorColumnOf(string column) => column + ".embedding";

    /// <summary>
    /// The column the cosine of two encoded columns' vectors is read from, <c>&lt;first&gt;~&lt;second&gt;.cosine</c> — added for
    /// every pair of columns encoded by one encoder, in declaration order: how close two texts are in meaning is what a
    /// pair of texts is asked about (similarity, duplicates, a question and its answer), and no trainer derives it from
    /// the two vectors' numbers on its own.
    /// </summary>
    public static string CosineColumnOf(string first, string second) => $"{first}~{second}.cosine";

    /// <summary>Why <paramref name="task"/> cannot read a text column through an encoder, or <see langword="null"/>.</summary>
    public static string? Unsupported(string task) => Tasks.Contains(task, StringComparer.OrdinalIgnoreCase)
        ? null
        : $"A text encoder feeds the trainer search's feature columns ({string.Join(", ", Tasks)}); '{task}' does not read them.";

    /// <summary>The encodings beside <paramref name="modelPath"/>, or <see langword="null"/> when the model reads no encoded column.</summary>
    public static TextEncodings? ReadBeside(string modelPath)
    {
        var path = PathBeside(modelPath);
        return File.Exists(path) ? JsonSerializer.Deserialize<TextEncodings>(File.ReadAllText(path), Options) : null;
    }

    /// <summary>Writes these encodings beside <paramref name="modelPath"/>.</summary>
    public void WriteBeside(string modelPath) => File.WriteAllText(PathBeside(modelPath), JsonSerializer.Serialize(this, Options));

    /// <summary>Where the encodings beside <paramref name="modelPath"/> live.</summary>
    public static string PathBeside(string modelPath)
        => Path.Combine(Path.GetDirectoryName(Path.GetFullPath(modelPath)) ?? ".", ExperimentLayout.TextEncodersFileName);

    /// <summary>
    /// <paramref name="model"/> with the encodings beside <paramref name="modelPath"/> in front of it, when there are
    /// any — the encoders opened from <paramref name="pool"/> (<see cref="EmbedderPool.Shared"/> when not given).
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// An encoder the model was trained with is not installed, is another (other weights, tokenizer or preprocessing),
    /// or cannot run here — the message says which and what to install.
    /// </exception>
    public static ITransformer ApplyBeside(MLContext mlContext, ITransformer model, string modelPath, EmbedderPool? pool = null)
    {
        if (ReadBeside(modelPath) is not { Columns.Count: > 0 } encodings)
            return model;
        var layer = encodings.Open(mlContext, pool ?? EmbedderPool.Shared);
        return new TransformerChain<ITransformer>(layer, model);
    }

    /// <summary>
    /// The encoding of these columns, from the encoders in <paramref name="pool"/> — each checked to be the one recorded.
    /// </summary>
    /// <exception cref="InvalidOperationException">An encoder is missing, another, or cannot run here.</exception>
    public TextEncodingLayer Open(MLContext mlContext, EmbedderPool pool)
    {
        ArgumentNullException.ThrowIfNull(pool);
        var opened = new List<(string, TextEmbedder)>();
        foreach (var column in Columns)
        {
            var (embedder, refusal) = pool.GetText(column.Encoder.Id);
            if (embedder is null)
                throw new InvalidOperationException($"Column '{column.Column}' was encoded by text encoder '{column.Encoder.Id}' for this model: {refusal}");
            if (Mismatch(column.Encoder, EncoderIdentity.Of(embedder.Encoder)) is { } mismatch)
                throw new InvalidOperationException(
                    $"Column '{column.Column}' was encoded by text encoder '{column.Encoder.Id}' with {mismatch}; the model reads only " +
                    $"those vectors. Install the encoder it was trained with, or retrain.");
            opened.Add((column.Column, embedder));
        }
        return new TextEncodingLayer(mlContext, opened);
    }

    // What differs between the encoder a model was trained with and the one installed, or null when they make the same vectors.
    private static string? Mismatch(EncoderIdentity trained, EncoderIdentity installed)
    {
        if (!string.Equals(trained.Sha256, installed.Sha256, StringComparison.OrdinalIgnoreCase))
            return $"weights {Short(trained.Sha256)}, but the installed one has {Short(installed.Sha256)}";
        if (!string.Equals(trained.TokenizerSha256, installed.TokenizerSha256, StringComparison.OrdinalIgnoreCase))
            return $"tokenizer {Short(trained.TokenizerSha256)}, but the installed one has {Short(installed.TokenizerSha256)}";
        if (trained.Preprocessor != installed.Preprocessor)
            return $"text preprocessing {trained.Preprocessor}, but this MLoop has {installed.Preprocessor}";
        return trained.Dimension != installed.Dimension
            ? $"{trained.Dimension} dimensions, but the installed one gives {installed.Dimension}"
            : null;
    }

    private static string Short(string? sha) => sha is { Length: >= 12 } ? sha[..12] : sha ?? "none";
}

/// <summary>A text column a model reads as the sentence vectors of <see cref="Encoder"/>.</summary>
public sealed record EncodedTextColumn(string Column, EncoderIdentity Encoder);

/// <summary>
/// Adds each encoded text column's sentence vector (<c>Vector&lt;Single, dimension&gt;</c>, named
/// <see cref="TextEncodings.VectorColumnOf"/>) beside it — the step in front of a model that reads encoded columns. A data view is encoded in batches: its texts are read
/// once and embedded together, then looked up row by row; a row scored alone (a prediction engine) is embedded alone.
/// Not saved with a model: it is put in front of one when it is loaded (<see cref="TextEncodings.ApplyBeside"/>), from
/// the encoders installed then.
/// </summary>
public sealed class TextEncodingLayer : ITransformer
{
    // Texts kept per column, at most: a view's texts are embedded together and then looked up, which a long-running
    // server scoring ever new texts must not grow without end — past this, a text is embedded where it is read.
    private const int MaxRemembered = 100_000;
    private const int BatchSize = 256;

    private readonly IReadOnlyList<Column> _columns;
    private readonly ITransformer _chain;

    internal TextEncodingLayer(MLContext mlContext, IReadOnlyList<(string Name, TextEmbedder Embedder)> columns)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        _columns = [.. columns.Select(c => new Column(c.Name, c.Embedder))];
        // Vectors in one space are compared: each pair of columns of one encoder gets its cosine — the dot product, as
        // the vectors are unit length.
        var pairs = from i in Enumerable.Range(0, _columns.Count)
                    from j in Enumerable.Range(i + 1, _columns.Count - i - 1)
                    where ReferenceEquals(_columns[i].Embedder, _columns[j].Embedder)
                    select Cosine(mlContext, _columns[i], _columns[j]);
        _chain = new TransformerChain<ITransformer>([.. _columns.Select(c => c.Mapping(mlContext)), .. pairs]);
    }

    private static ITransformer Cosine(MLContext mlContext, Column first, Column second)
    {
        var vector = new VectorDataViewType(NumberDataViewType.Single, first.Embedder.Encoder.Dimension);
        var input = SchemaDefinition.Create(typeof(PairIn));
        var a = input.Single(c => c.MemberName == nameof(PairIn.A));
        var b = input.Single(c => c.MemberName == nameof(PairIn.B));
        (a.ColumnName, a.ColumnType) = (TextEncodings.VectorColumnOf(first.Name), vector);
        (b.ColumnName, b.ColumnType) = (TextEncodings.VectorColumnOf(second.Name), vector);
        var output = SchemaDefinition.Create(typeof(CosineOut));
        output[0].ColumnName = TextEncodings.CosineColumnOf(first.Name, second.Name);
        return mlContext.Transforms.CustomMapping<PairIn, CosineOut>(
                (row, result) =>
                {
                    var dot = 0.0;
                    for (var d = 0; d < row.A.Length && d < row.B.Length; d++)
                        dot += (double)row.A[d] * row.B[d];
                    result.Cosine = (float)dot;
                }, contractName: null, input, output)
            .Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<PairIn>(), input));
    }

    /// <summary>The columns encoded, each with the encoder's identity — what <see cref="TextEncodings"/> records.</summary>
    public TextEncodings Encodings => new([.. _columns.Select(c => new EncodedTextColumn(c.Name, EncoderIdentity.Of(c.Embedder.Encoder)))]);

    public bool IsRowToRowMapper => true;

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => _chain.GetOutputSchema(inputSchema);

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => _chain.GetRowToRowMapper(inputSchema);

    /// <summary>Encodes <paramref name="input"/>: its texts are embedded in batches first, then each row reads its vector.</summary>
    public IDataView Transform(IDataView input)
    {
        ArgumentNullException.ThrowIfNull(input);
        foreach (var column in _columns)
            column.Remember(input);
        return _chain.Transform(input);
    }

    void ICanSaveModel.Save(ModelSaveContext ctx) => throw new NotSupportedException(
        $"Text encodings are put in front of a model when it is loaded, from {ExperimentLayout.TextEncodersFileName}; they are not saved in it.");

    private sealed class Column(string name, TextEmbedder embedder)
    {
        private readonly ConcurrentDictionary<string, float[]> _vectors = new(StringComparer.Ordinal);

        public string Name { get; } = name;

        public TextEmbedder Embedder { get; } = embedder;

        // The vector of every text in the view's column that is not yet known, embedded in batches.
        public void Remember(IDataView view)
        {
            var unknown = new HashSet<string>(StringComparer.Ordinal);
            foreach (var text in view.GetColumn<ReadOnlyMemory<char>>(Name))
            {
                var s = text.ToString();
                if (!_vectors.ContainsKey(s) && _vectors.Count + unknown.Count < MaxRemembered)
                    unknown.Add(s);
            }
            foreach (var batch in unknown.Chunk(BatchSize))
            {
                var vectors = Embedder.Embed(batch).Vectors;
                for (var i = 0; i < batch.Length; i++)
                    _vectors[batch[i]] = vectors[i];
            }
        }

        public ITransformer Mapping(MLContext mlContext)
        {
            var dimension = Embedder.Encoder.Dimension;
            var input = SchemaDefinition.Create(typeof(TextIn));
            input[0].ColumnName = Name;
            var output = SchemaDefinition.Create(typeof(VectorOut));
            output[0].ColumnName = TextEncodings.VectorColumnOf(Name);
            output[0].ColumnType = new VectorDataViewType(NumberDataViewType.Single, dimension);
            return mlContext.Transforms.CustomMapping<TextIn, VectorOut>(
                    (row, result) => result.Vector = Vector(row.Text ?? ""), contractName: null, input, output)
                .Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<TextIn>(), input));
        }

        private float[] Vector(string text) => _vectors.TryGetValue(text, out var vector) ? vector : Embedder.Embed([text]).Vectors[0];
    }

    private sealed class PairIn
    {
        public float[] A { get; set; } = [];

        public float[] B { get; set; } = [];
    }

    private sealed class CosineOut
    {
        public float Cosine { get; set; }
    }

    private sealed class TextIn
    {
        public string? Text { get; set; }
    }

    private sealed class VectorOut
    {
        public float[] Vector { get; set; } = [];
    }
}
