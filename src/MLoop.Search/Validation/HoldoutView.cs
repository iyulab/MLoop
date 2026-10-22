using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Search.Validation;

/// <summary>
/// The source data with one more column, <see cref="ColumnName"/>: 1 for the rows of the validated
/// part, 0 for the rest — what <c>FilterRowsByColumn</c> splits a fold on. The value comes from the
/// row's position, which is why the view never shuffles and gives one cursor.
/// </summary>
internal sealed class HoldoutView : IDataView
{
    public const string ColumnName = "__MLoopHoldout";

    private readonly IDataView _source;
    private readonly IReadOnlyList<int> _foldOf;
    private readonly int _part;

    public HoldoutView(IDataView source, IReadOnlyList<int> foldOf, int part)
    {
        _source = source;
        _foldOf = foldOf;
        _part = part;
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(source.Schema);
        builder.AddColumn(ColumnName, NumberDataViewType.Single);
        Schema = builder.ToSchema();
    }

    public bool CanShuffle => false;

    public DataViewSchema Schema { get; }

    public long? GetRowCount() => _source.GetRowCount();

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
    {
        var needed = columnsNeeded.ToList();
        var sourceColumns = needed.Where(c => c.Index < _source.Schema.Count).Select(c => _source.Schema[c.Index]);
        return new Cursor(this, _source.GetRowCursor(sourceColumns), needed.Any(c => c.Index == _source.Schema.Count));
    }

    public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
        => [GetRowCursor(columnsNeeded, rand)];

    private sealed class Cursor(HoldoutView view, DataViewRowCursor source, bool holdoutActive) : DataViewRowCursor
    {
        public override DataViewSchema Schema => view.Schema;
        public override long Position => source.Position;
        public override long Batch => source.Batch;

        public override bool IsColumnActive(DataViewSchema.Column column)
            => column.Index == view._source.Schema.Count ? holdoutActive : source.IsColumnActive(view._source.Schema[column.Index]);

        public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
        {
            if (column.Index != view._source.Schema.Count)
                return source.GetGetter<TValue>(view._source.Schema[column.Index]);
            ValueGetter<float> getter = (ref float value) => value = view._foldOf[checked((int)source.Position)] == view._part ? 1f : 0f;
            return (ValueGetter<TValue>)(Delegate)getter;
        }

        public override ValueGetter<DataViewRowId> GetIdGetter() => source.GetIdGetter();

        public override bool MoveNext() => source.MoveNext();

        protected override void Dispose(bool disposing)
        {
            if (disposing)
                source.Dispose();
            base.Dispose(disposing);
        }
    }
}
