using Microsoft.ML;
using Microsoft.ML.TorchSharp;
using MLoop.Core.Data;
using MLoop.Core.Models;

namespace MLoop.Core.AutoML;

/// <summary>
/// Deep-learning task handlers (TensorFlow/TorchSharp-backed): image-classification,
/// text-classification, sentence-similarity, ner, object-detection, question-answering.
///
/// Kept in this partial — separate from the classic-ML.NET tabular handlers — so the
/// Microsoft.ML.TorchSharp / Microsoft.ML.Vision usings stay confined to one file. This is
/// stage 1 of upstream-007 (tabular-slim MLoop.Core): the managed DL assemblies are still
/// compile-time references of MLoop.Core, but extracting them into an optional
/// MLoop.Core.DeepLearning package is now a mechanical move of this file plus
/// ObjectDetectionEvaluator, ImageDirectoryLoader, and the Torch/TensorFlow
/// RuntimeDefinitions. Native runtimes are already task-gated at run time by
/// RuntimeManager.EnsureRuntimeForTask.
/// </summary>
public partial class AutoMLRunner
{
    private async Task<AutoMLResult> RunImageClassificationAsync(
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            _logger.Info($"Image classification: label='{config.LabelColumn}', TensorFlow transfer learning");

            // The ImageClassification trainer requires raw image bytes as its feature
            // column. ImageDirectoryLoader produces an "ImagePath" string column, so
            // LoadRawImageBytes reads each file into a VarVector<byte> before fitting.
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey("Label", config.LabelColumn)
                .Append(_mlContext.Transforms.LoadRawImageBytes(
                    outputColumnName: "ImageBytes", imageFolder: null, inputColumnName: "ImagePath"))
                .Append(_mlContext.MulticlassClassification.Trainers.ImageClassification(
                    featureColumnName: "ImageBytes", labelColumnName: "Label"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            progress?.Report(new TrainingProgress { TrialNumber = 1, TrainerName = "ImageClassification (TF)", MetricName = "accuracy", Metric = 0, ElapsedSeconds = 0 });

            var model = pipeline.Fit(trainSet);
            var predictions = model.Transform(testSet);
            var metrics = _mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label");

            return new AutoMLResult
            {
                BestTrainer = "ImageClassification (TensorFlow)",
                Model = model,
                Metrics = new Dictionary<string, double>
                {
                    ["accuracy"] = NanSafe(metrics.MacroAccuracy),
                    ["micro_accuracy"] = NanSafe(metrics.MicroAccuracy),
                    ["log_loss"] = NanSafe(metrics.LogLoss)
                },
                RowCount = trainSet.GetRowCount() ?? 0
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<AutoMLResult> RunTextClassificationAsync(
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            var textCol = FindFirstTextColumn(trainSet.Schema, config.LabelColumn)
                ?? throw new InvalidOperationException("No text column found for text classification.");

            _logger.Info($"Text classification: text='{textCol}', label='{config.LabelColumn}'");

            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey("Label", config.LabelColumn)
                .Append(_mlContext.MulticlassClassification.Trainers.TextClassification(
                    labelColumnName: "Label", sentence1ColumnName: textCol))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            progress?.Report(new TrainingProgress { TrialNumber = 1, TrainerName = "TextClassification (NAS-BERT)", MetricName = "accuracy", Metric = 0, ElapsedSeconds = 0 });

            var model = pipeline.Fit(trainSet);
            var predictions = model.Transform(testSet);
            var metrics = _mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label");

            return new AutoMLResult
            {
                BestTrainer = "TextClassification (NAS-BERT)",
                Model = model,
                Metrics = new Dictionary<string, double>
                {
                    ["accuracy"] = NanSafe(metrics.MacroAccuracy),
                    ["micro_accuracy"] = NanSafe(metrics.MicroAccuracy),
                    ["log_loss"] = NanSafe(metrics.LogLoss)
                },
                RowCount = trainSet.GetRowCount() ?? 0
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<AutoMLResult> RunSentenceSimilarityAsync(
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            var textCols = FindTextColumns(trainSet.Schema, config.LabelColumn, 2);
            if (textCols.Count < 2)
                throw new InvalidOperationException("Sentence similarity requires at least two text columns.");

            _logger.Info($"Sentence similarity: s1='{textCols[0]}', s2='{textCols[1]}', label='{config.LabelColumn}'");

            var pipeline = _mlContext.Regression.Trainers.SentenceSimilarity(
                labelColumnName: config.LabelColumn,
                sentence1ColumnName: textCols[0],
                sentence2ColumnName: textCols[1]);

            progress?.Report(new TrainingProgress { TrialNumber = 1, TrainerName = "SentenceSimilarity (NAS-BERT)", MetricName = "r_squared", Metric = 0, ElapsedSeconds = 0 });

            var model = pipeline.Fit(trainSet);
            var predictions = model.Transform(testSet);
            var metrics = _mlContext.Regression.Evaluate(predictions, labelColumnName: config.LabelColumn);

            return new AutoMLResult
            {
                BestTrainer = "SentenceSimilarity (NAS-BERT)",
                Model = model,
                Metrics = new Dictionary<string, double>
                {
                    ["r_squared"] = NanSafe(metrics.RSquared),
                    ["rmse"] = NanSafe(metrics.RootMeanSquaredError),
                    ["mae"] = NanSafe(metrics.MeanAbsoluteError)
                },
                RowCount = trainSet.GetRowCount() ?? 0
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<AutoMLResult> RunNerAsync(
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            var textCol = FindFirstTextColumn(trainSet.Schema, config.LabelColumn)
                ?? throw new InvalidOperationException("No text column found for NER.");

            _logger.Info($"NER: text='{textCol}', label='{config.LabelColumn}'");

            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey("Label", config.LabelColumn)
                .Append(_mlContext.MulticlassClassification.Trainers.NamedEntityRecognition(
                    labelColumnName: "Label", sentence1ColumnName: textCol))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            progress?.Report(new TrainingProgress { TrialNumber = 1, TrainerName = "NER (NAS-BERT)", MetricName = "accuracy", Metric = 0, ElapsedSeconds = 0 });

            var model = pipeline.Fit(trainSet);
            var predictions = model.Transform(testSet);
            var metrics = _mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label");

            return new AutoMLResult
            {
                BestTrainer = "NER (NAS-BERT)",
                Model = model,
                Metrics = new Dictionary<string, double>
                {
                    ["accuracy"] = NanSafe(metrics.MacroAccuracy),
                    ["micro_accuracy"] = NanSafe(metrics.MicroAccuracy)
                },
                RowCount = trainSet.GetRowCount() ?? 0
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<AutoMLResult> RunObjectDetectionAsync(
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            var labelColumn = string.IsNullOrWhiteSpace(config.LabelColumn)
                ? CocoDataLoader.DefaultLabelColumn
                : config.LabelColumn;

            _logger.Info($"Object detection: label='{labelColumn}', AutoFormerV2 transfer learning");

            // CocoDataLoader produces three columns: ImagePath (string), the label vector
            // (VBuffer<string>, one class name per object), and BoundingBoxes (VBuffer<float>,
            // four values per object in x0 y0 x1 y1 order). The AutoFormerV2 ObjectDetection
            // trainer requires the image as an MLImage, the label as a vector of keys, and the
            // bounding-box float vector as-is — so LoadImages converts the path and
            // MapValueToKey converts the label vector before fitting.
            var pipeline = _mlContext.Transforms.LoadImages(
                    outputColumnName: "Image", imageFolder: string.Empty, inputColumnName: CocoDataLoader.ImagePathColumn)
                .Append(_mlContext.Transforms.Conversion.MapValueToKey(
                    outputColumnName: "LabelKey", inputColumnName: labelColumn))
                .Append(_mlContext.MulticlassClassification.Trainers.ObjectDetection(
                    labelColumnName: "LabelKey",
                    boundingBoxColumnName: CocoDataLoader.BoundingBoxColumn,
                    imageColumnName: "Image"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue(
                    outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel"));

            // D27: real-data OD training intermittently dies with a native access violation
            // (0xC0000005) inside libtorch, crash location varying between runs (Linear_forward,
            // Tensor.backward). Upstream research (dotnet/TorchSharp#1292) traces this class of
            // crash to native heap corruption whose exact failure point shifts with memory/thread
            // pressure — forcing libtorch to a single thread removes that pressure source. This is
            // a defensive mitigation, not a confirmed fix (unverified under this investigation's
            // resource-constrained environment — see D27 issue); it carries no downside beyond
            // slower CPU training, so it is applied unconditionally rather than gated on success.
            TorchSharp.torch.set_num_threads(1);

            progress?.Report(new TrainingProgress { TrialNumber = 1, TrainerName = "ObjectDetection (AutoFormerV2)", MetricName = "accuracy", Metric = 0, ElapsedSeconds = 0 });

            var model = pipeline.Fit(trainSet);
            var predictions = model.Transform(testSet);

            return new AutoMLResult
            {
                BestTrainer = "ObjectDetection (AutoFormerV2)",
                Model = model,
                Metrics = new Dictionary<string, double>(),
                RowCount = trainSet.GetRowCount() ?? 0
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<AutoMLResult> RunQuestionAnsweringAsync(
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            var textCols = FindTextColumns(trainSet.Schema, config.LabelColumn, 2);
            var contextCol = textCols.Count > 0 ? textCols[0] : throw new InvalidOperationException("No context column found.");
            var questionCol = textCols.Count > 1 ? textCols[1] : contextCol;

            _logger.Info($"Question answering: context='{contextCol}', question='{questionCol}', answer='{config.LabelColumn}'");

            var pipeline = _mlContext.MulticlassClassification.Trainers.QuestionAnswer(
                contextColumnName: contextCol,
                questionColumnName: questionCol);

            progress?.Report(new TrainingProgress { TrialNumber = 1, TrainerName = "QA (NAS-BERT)", MetricName = "accuracy", Metric = 0, ElapsedSeconds = 0 });

            var model = pipeline.Fit(trainSet);

            return new AutoMLResult
            {
                BestTrainer = "QA (NAS-BERT)",
                Model = model,
                Metrics = new Dictionary<string, double>(),
                RowCount = trainSet.GetRowCount() ?? 0
            };
        }, cancellationToken).ConfigureAwait(false);
    }
}
