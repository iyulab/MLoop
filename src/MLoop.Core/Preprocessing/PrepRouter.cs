using Microsoft.ML;
using MLoop.Core.AutoML;

namespace MLoop.Core.Preprocessing;

public record PrepRouteResult(
    IEstimator<ITransformer>? PreFeaturizer,
    List<PrepStep> CsvSteps,
    List<string> Warnings);

/// <summary>
/// mloop.yaml prep 스텝을 누수 안전 경로로 라우팅한다.
/// 통계 fit 변환은 ML.NET preFeaturizer(fold-내 fit)로, 나머지는 CSV prep 단계로 보낸다.
/// 매핑 불가한 통계 변환(median fill, rolling, resample)은 CSV로 보내되 경고를 수집한다.
/// </summary>
public class PrepRouter
{
    public PrepRouteResult Route(MLContext ctx, IReadOnlyList<PrepStep> steps)
    {
        var preFeaturizer = new PrepFeaturizerBuilder().Build(ctx, steps);
        var csvSteps = new List<PrepStep>();
        var warnings = new List<string>();

        foreach (var step in steps)
        {
            var category = PrepStepClassifier.Classify(step);
            if (category == PrepCategory.PreFeaturizer)
                continue; // preFeaturizer가 흡수 — CSV에서 제외

            csvSteps.Add(step);
            if (category == PrepCategory.UnsupportedLeakageWarn)
                warnings.Add(PrepStepClassifier.LeakageWarning(step));
        }

        return new PrepRouteResult(preFeaturizer, csvSteps, warnings);
    }
}
