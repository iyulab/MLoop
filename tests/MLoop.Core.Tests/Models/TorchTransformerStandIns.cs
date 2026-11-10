using Microsoft.ML;
using Microsoft.ML.Data;

// Stand-ins that carry the type names MLoop reads ML.NET's TorchSharp transformers by (MLoop.Core does not reference TorchSharp) and
// the layout it reads their options from: a non-public `Options` field on a base type, holding the column names the model was fitted on.
namespace Microsoft.ML.TorchSharp.NasBert
{
    public class StandInTransformerBase
    {
        internal object Options = new StandInOptions { Sentence1ColumnName = "Sentence" };
    }

    public sealed class StandInOptions
    {
        public string Sentence1ColumnName = "";
    }

    /// <summary>Named as ML.NET's NER transformer; reads its sentences from <c>Options.Sentence1ColumnName</c>.</summary>
    public sealed class NerTransformer : StandInTransformerBase, ITransformer
    {
        public NerTransformer(object? options = null)
        {
            if (options is not null) Options = options;
        }

        public bool IsRowToRowMapper => false;

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => inputSchema;

        public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => throw new NotSupportedException();

        public IDataView Transform(IDataView input) => input;

        void ICanSaveModel.Save(ModelSaveContext ctx) => throw new NotSupportedException();
    }
}

namespace Microsoft.ML.TorchSharp.Roberta
{
    public class StandInTransformerBase
    {
        internal object Options = new StandInOptions { ContextColumnName = "Passage" };
    }

    public sealed class StandInOptions
    {
        public string ContextColumnName = "";
    }

    /// <summary>Named as ML.NET's question-answering transformer; reads its passages from <c>Options.ContextColumnName</c>.</summary>
    public sealed class QATransformer : StandInTransformerBase, ITransformer
    {
        public QATransformer(object? options = null)
        {
            if (options is not null) Options = options;
        }

        public bool IsRowToRowMapper => false;

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => inputSchema;

        public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => throw new NotSupportedException();

        public IDataView Transform(IDataView input) => input;

        void ICanSaveModel.Save(ModelSaveContext ctx) => throw new NotSupportedException();
    }
}
