using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Core.Models;

/// <summary>
/// A model followed by a step that rewrites its multiclass <c>Score</c> row by row — the class probabilities a
/// calibration tempers, the logits a classifier gives in place of probabilities. Applied when the model is loaded,
/// never saved: the step is not a component ML.NET can deserialize.
/// </summary>
public static class ScoreMapping
{
    /// <summary>
    /// <paramref name="model"/> with every row's <c>Score</c> vector replaced by <paramref name="map"/> of it. The new
    /// column keeps the old one's class-name annotations (<c>SlotNames</c>, <c>TrainingLabelValues</c>):
    /// <see cref="MulticlassScoreVocabulary"/> names the classes from them, and a replacement without them would leave
    /// every probability keyed by position. A model without a vector <c>Score</c> is answered unchanged.
    /// </summary>
    public static ITransformer Apply(MLContext mlContext, ITransformer model, Func<float[], float[]> map)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(map);
        return new MappedModel(mlContext, model, map);
    }

    // The step's Score width and annotations come from the model's output on the schema it is actually given — a
    // saved model need not carry its input schema — so the chain is built on first use for a schema and kept for it.
    private sealed class MappedModel(MLContext mlContext, ITransformer model, Func<float[], float[]> map) : ITransformer
    {
        private readonly object _gate = new();
        private (DataViewSchema Schema, ITransformer Chain)? _built;

        public bool IsRowToRowMapper => model.IsRowToRowMapper;

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => ChainFor(inputSchema).GetOutputSchema(inputSchema);

        public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => ChainFor(inputSchema).GetRowToRowMapper(inputSchema);

        public IDataView Transform(IDataView input) => ChainFor(input.Schema).Transform(input);

        void ICanSaveModel.Save(ModelSaveContext ctx) =>
            throw new NotSupportedException("A model whose scores are mapped at load is saved as the model alone.");

        private ITransformer ChainFor(DataViewSchema schema)
        {
            lock (_gate)
            {
                if (_built is { } built && ReferenceEquals(built.Schema, schema))
                    return built.Chain;
                var chain = Build(schema);
                _built = (schema, chain);
                return chain;
            }
        }

        private ITransformer Build(DataViewSchema inputSchema)
        {
            var score = model.GetOutputSchema(inputSchema).GetColumnOrNull("Score");
            if (score?.Type is not VectorDataViewType { Size: > 0 } vector || vector.ItemType != NumberDataViewType.Single)
                return model;
            var width = vector.Size;

            var input = SchemaDefinition.Create(typeof(ScoreVector));
            input["Score"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, width);
            var output = SchemaDefinition.Create(typeof(ScoreVector));
            output["Score"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, width);
            foreach (var annotation in score.Value.Annotations.Schema)
                CopyAnnotation(score.Value, annotation, output["Score"]);

            var step = mlContext.Transforms.CustomMapping<ScoreVector, ScoreVector>(
                    (row, mapped) => mapped.Score = map(row.Score),
                    contractName: null, inputSchemaDefinition: input, outputSchemaDefinition: output)
                .Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<ScoreVector>(), input));
            return new TransformerChain<ITransformer>(model, step);
        }
    }

    private static void CopyAnnotation(DataViewSchema.Column source, DataViewSchema.Column annotation, SchemaDefinition.Column target)
    {
        switch (annotation.Type)
        {
            case VectorDataViewType { ItemType: TextDataViewType }:
                {
                    VBuffer<ReadOnlyMemory<char>> value = default;
                    source.Annotations.GetValue(annotation.Name, ref value);
                    target.AddAnnotation(annotation.Name, value, annotation.Type);
                    break;
                }
            case VectorDataViewType { ItemType: NumberDataViewType n } when n == NumberDataViewType.Single:
                {
                    VBuffer<float> value = default;
                    source.Annotations.GetValue(annotation.Name, ref value);
                    target.AddAnnotation(annotation.Name, value, annotation.Type);
                    break;
                }
                // Other annotations (score kind, column set) describe the trainer's output; the vocabulary
                // does not read them, and a scored column without them still scores.
        }
    }

    /// <summary>The row a mapping reads and writes.</summary>
    public sealed class ScoreVector
    {
        public float[] Score { get; set; } = [];
    }
}

/// <summary>
/// Whether a classifier's <c>Score</c> holds class probabilities or logits, and the probabilities when it holds
/// logits. ML.NET's own multiclass trainers give probabilities; the NAS-BERT text classifier (TorchSharp) gives the
/// network's logits — negative values that do not sum to one — under the same column name. Read as probabilities they
/// were reported as <c>probabilities</c> and <c>confidence</c> (0.03 for a confident answer), and the evaluator's log
/// loss over them was meaningless (15.7 at accuracy 1.0).
/// </summary>
public static class ClassScores
{
    // Known by the model itself, not by a task name: the serving cache and file prediction load without one.
    private const string LogitTransformer = "Microsoft.ML.TorchSharp.NasBert.TextClassificationTransformer";

    /// <summary>Whether <paramref name="model"/> (or a transformer in its chain) scores with logits.</summary>
    public static bool AreLogits(ITransformer model) => model switch
    {
        null => false,
        _ when model.GetType().FullName == LogitTransformer => true,
        IEnumerable<ITransformer> chain => chain.Any(AreLogits),
        _ => false,
    };

    /// <summary><paramref name="model"/> scoring with class probabilities: softmax over its logits, or the model itself.</summary>
    public static ITransformer AsProbabilities(MLContext mlContext, ITransformer model)
        => AreLogits(model) ? ScoreMapping.Apply(mlContext, model, Softmax) : model;

    /// <summary>The probabilities of <paramref name="logits"/> — shifted by the largest, so no exponent overflows.</summary>
    public static float[] Softmax(float[] logits)
    {
        if (logits.Length == 0)
            return logits;
        var max = logits.Max();
        var exp = logits.Select(z => Math.Exp(z - max)).ToArray();
        var total = exp.Sum();
        return [.. exp.Select(e => (float)(e / total))];
    }
}
