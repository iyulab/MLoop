using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.AutoML;

namespace MLoop.Core.Tests.AutoML;

/// <summary>
/// Pins the shape of the manual binary fallback (<see cref="AutoMLRunner.BuildManualBinaryPipeline"/>):
/// featurization is computed <b>once</b> and cached before the iterative trainer reads it.
/// </summary>
/// <remarks>
/// Without the cache checkpoint, SDCA re-ran the entire text-featurization chain from the source
/// rows on every pass — on a 50-row, twelve-text-column dataset the fallback took five and a half
/// minutes with no budget able to stop it. A wall-clock bound would measure the machine, so the
/// guard counts something load-independent instead: how many times training opened a cursor on the
/// source data. Each estimator's fit is one pass and the cache fill is one more; a pass per SDCA
/// iteration on top of that is what the bound rules out.
/// </remarks>
public class ManualBinaryPipelineTests
{
    private readonly MLContext _mlContext = new(seed: 42);

    [Fact]
    public void Fit_ReadsTheSourceABoundedNumberOfTimes()
    {
        var source = new CountingDataView(_mlContext.Data.LoadFromEnumerable(Rows(40)));
        var columnInfo = AutoMLRunner.BuildColumnInformation(source, "Label");
        Assert.NotNull(columnInfo);
        Assert.Equal(12, columnInfo!.TextColumnNames.Count);

        var pipeline = AutoMLRunner.BuildManualBinaryPipeline(_mlContext, source, "Label", columnInfo);
        var model = pipeline.Fit(source);

        // Measured: 37 opens with the checkpoint — twelve text featurizers, a concatenation and the
        // cache fill, each reading the source a few times (schema probe + data pass). Without it the
        // same fit did not finish in ten minutes: SDCA reaches the source on every pass. The bound
        // leaves room for an estimator or two more, not for a trainer pass.
        Assert.InRange(source.CursorOpens, 1, 60);

        // The model it produced is a real, calibrated classifier over the same schema.
        var predictions = model.Transform(source);
        Assert.NotNull(predictions.Schema.GetColumnOrNull("Probability"));
    }

    [Fact]
    public void Pipeline_HasNoTextColumns_StillTrains()
    {
        // The numeric-only branch of BuildFeaturePipeline (columnInfo == null): every non-label
        // column is a feature, and the same cache → SDCA tail applies.
        var data = _mlContext.Data.LoadFromEnumerable(Enumerable.Range(0, 40).Select(i => new NumericRow
        {
            A = i, B = i % 3, Label = i % 2 == 0,
        }));

        var model = AutoMLRunner.BuildManualBinaryPipeline(_mlContext, data, "Label", columnInfo: null).Fit(data);

        Assert.NotNull(model.Transform(data).Schema.GetColumnOrNull("PredictedLabel"));
    }

    private static IEnumerable<TextHeavyRow> Rows(int count)
    {
        var rng = new Random(7);
        string Word() => new[] { "alpha", "beta", "gamma", "delta", "none", "fiber", "dsl" }[rng.Next(7)];
        for (var i = 0; i < count; i++)
        {
            yield return new TextHeavyRow
            {
                Id = $"{1000 + i}-{Word().ToUpperInvariant()}",
                T1 = Word(), T2 = Word(), T3 = Word(), T4 = Word(), T5 = Word(), T6 = Word(),
                T7 = Word(), T8 = Word(), T9 = Word(), T10 = Word(), T11 = Word(),
                Amount = (float)rng.NextDouble() * 100,
                Label = i % 3 == 0,
            };
        }
    }

    private sealed class TextHeavyRow
    {
        public string Id { get; set; } = "";
        public string T1 { get; set; } = ""; public string T2 { get; set; } = "";
        public string T3 { get; set; } = ""; public string T4 { get; set; } = "";
        public string T5 { get; set; } = ""; public string T6 { get; set; } = "";
        public string T7 { get; set; } = ""; public string T8 { get; set; } = "";
        public string T9 { get; set; } = ""; public string T10 { get; set; } = "";
        public string T11 { get; set; } = "";
        public float Amount { get; set; }
        public bool Label { get; set; }
    }

    private sealed class NumericRow
    {
        public float A { get; set; }
        public float B { get; set; }
        public bool Label { get; set; }
    }

    /// <summary>Delegating <see cref="IDataView"/> that counts how often a cursor is opened on it.</summary>
    private sealed class CountingDataView(IDataView inner) : IDataView
    {
        private int _opens;

        public int CursorOpens => _opens;
        public bool CanShuffle => inner.CanShuffle;
        public DataViewSchema Schema => inner.Schema;
        public long? GetRowCount() => inner.GetRowCount();

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
        {
            System.Threading.Interlocked.Increment(ref _opens);
            return inner.GetRowCursor(columnsNeeded, rand);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
        {
            System.Threading.Interlocked.Increment(ref _opens);
            return inner.GetRowCursorSet(columnsNeeded, n, rand);
        }
    }
}
