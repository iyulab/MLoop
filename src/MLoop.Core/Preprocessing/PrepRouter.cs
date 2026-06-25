using Microsoft.ML;
using MLoop.Core.AutoML;

namespace MLoop.Core.Preprocessing;

public record PrepRouteResult(
    IEstimator<ITransformer>? PreFeaturizer,
    List<PrepStep> CsvSteps,
    List<string> Warnings,
    List<string> PreFeaturizerColumns);

/// <summary>
/// mloop.yaml prep 스텝을 누수 안전 경로로 라우팅한다.
/// 통계 fit 변환은 ML.NET preFeaturizer(fold-내 fit)로, 나머지는 CSV prep 단계로 보낸다.
/// 매핑 불가한 통계 변환(median fill, rolling, resample)은 CSV로 보내되 경고를 수집한다.
/// </summary>
public class PrepRouter
{
    /// <param name="ctx">ML.NET context used to build the preFeaturizer estimator.</param>
    /// <param name="steps">prep steps declared in mloop.yaml.</param>
    /// <param name="supportsPreFeaturizer">
    /// True only for task types whose AutoML Execute site consumes <c>config.PreFeaturizer</c>
    /// (binary/multiclass/regression — see <see cref="AutoML.AutoMLRunner.SupportsPreFeaturizer"/>).
    /// When false, statistical transforms cannot be applied via the (ignored) preFeaturizer, so they
    /// are kept in the CSV stage instead (still applied, but leaky → leakage warning emitted).
    /// This avoids a silent behavioral regression for scale-sensitive tasks like clustering/anomaly.
    /// </param>
    public PrepRouteResult Route(MLContext ctx, IReadOnlyList<PrepStep> steps, bool supportsPreFeaturizer = true)
    {
        var csvSteps = new List<PrepStep>();
        var warnings = new List<string>();

        foreach (var step in steps)
        {
            var category = PrepStepClassifier.Classify(step);

            if (category == PrepCategory.PreFeaturizer)
            {
                if (supportsPreFeaturizer)
                    continue; // preFeaturizer가 흡수 — CSV에서 제외

                // Task ignores preFeaturizer: keep in CSV (applied, but leaky) + warn instead of dropping.
                csvSteps.Add(step);
                warnings.Add(PrepStepClassifier.LeakageWarning(step));
                continue;
            }

            csvSteps.Add(step);
            if (category == PrepCategory.UnsupportedLeakageWarn)
                warnings.Add(PrepStepClassifier.LeakageWarning(step));
        }

        // Only build/expose the preFeaturizer for tasks that actually consume it.
        var preFeaturizer = supportsPreFeaturizer ? new PrepFeaturizerBuilder().Build(ctx, steps) : null;
        var preFeaturizerColumns = supportsPreFeaturizer
            ? PrepFeaturizerBuilder.ResolvePreFeaturizerColumns(steps).ToList()
            : new List<string>();

        return new PrepRouteResult(preFeaturizer, csvSteps, warnings, preFeaturizerColumns);
    }
}
