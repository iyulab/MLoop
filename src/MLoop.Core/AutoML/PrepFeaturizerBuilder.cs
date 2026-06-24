using Microsoft.ML;
using Microsoft.ML.Transforms;
using MLoop.Core.Preprocessing;

namespace MLoop.Core.AutoML;

/// <summary>
/// mloop.yaml prep의 통계 fit 변환(normalize/scale/fill-missing mean)을
/// ML.NET preFeaturizer EstimatorChain으로 변환한다. ML.NET AutoML이 이 estimator를
/// fold의 train split에서만 fit하므로 누수가 발생하지 않는다.
/// PreFeaturizer 분류가 아닌 스텝은 무시한다(호출자가 CSV 단계에서 처리).
/// </summary>
public class PrepFeaturizerBuilder
{
    public IEstimator<ITransformer>? Build(MLContext ctx, IReadOnlyList<PrepStep> steps)
    {
        IEstimator<ITransformer>? chain = null;

        foreach (var step in steps)
        {
            if (PrepStepClassifier.Classify(step) != PrepCategory.PreFeaturizer)
                continue;

            foreach (var col in ResolveColumns(step))
            {
                var est = BuildOne(ctx, step, col);
                chain = chain is null ? est : chain.Append(est);
            }
        }

        return chain;
    }

    private static IEstimator<ITransformer> BuildOne(MLContext ctx, PrepStep step, string col)
    {
        var type = step.Type.ToLowerInvariant().Replace('_', '-');
        var method = (step.Method ?? "min-max").ToLowerInvariant().Replace('_', '-');

        if (type is "normalize" or "scale")
        {
            return method switch
            {
                "z-score" => ctx.Transforms.NormalizeMeanVariance(col, col),
                "robust" => ctx.Transforms.NormalizeRobustScaling(col, col),
                _ => ctx.Transforms.NormalizeMinMax(col, col) // min-max 기본
            };
        }

        // fill-missing mean (median/constant 등은 Classify에서 걸러져 여기 도달 안 함)
        // ReplaceMissingValues는 Single 입력 필요 → ConvertType 선행 후 대치
        return ctx.Transforms.Conversion.ConvertType(col, col, Microsoft.ML.Data.DataKind.Single)
            .Append(ctx.Transforms.ReplaceMissingValues(
                col, col, MissingValueReplacingEstimator.ReplacementMode.Mean));
    }

    private static IEnumerable<string> ResolveColumns(PrepStep step)
    {
        if (step.Columns is { Count: > 0 }) return step.Columns;
        if (!string.IsNullOrWhiteSpace(step.Column)) return new[] { step.Column! };
        return Array.Empty<string>();
    }
}
