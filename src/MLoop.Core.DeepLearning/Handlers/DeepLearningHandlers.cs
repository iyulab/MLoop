using Microsoft.ML;
using Microsoft.ML.TorchSharp;
using MLoop.Core.AutoML;
using MLoop.Core.Data;
using MLoop.Core.Models;

namespace MLoop.Core.DeepLearning;

/// <summary>
/// Deep-learning task handlers (TensorFlow/TorchSharp-backed): image-classification,
/// text-classification, sentence-similarity, ner, object-detection, question-answering.
///
/// Moved out of <c>MLoop.Core.AutoML.AutoMLRunner</c> (upstream-007 stage 2, task 3) into this
/// optional <c>MLoop.Core.DeepLearning</c> assembly, so the Microsoft.ML.TorchSharp /
/// Microsoft.ML.Vision usings — and their heavy native runtime dependencies — no longer live in
/// MLoop.Core. Invoked via <see cref="DeepLearningModule"/>, which is registered with
/// <see cref="DeepLearningRegistry"/> by consumers (MLoop.CLI / MLoop.API) that opt into DL support.
/// </summary>
internal static class DeepLearningHandlers
{
    public static async Task<AutoMLResult> RunImageClassificationAsync(
        MLContext mlContext, Action<string> log,
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            log($"Image classification: label='{config.LabelColumn}', TensorFlow transfer learning");

            // The ImageClassification trainer requires raw image bytes as its feature
            // column. ImageDirectoryLoader produces an "ImagePath" string column, so
            // LoadRawImageBytes reads each file into a VarVector<byte> before fitting.
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", config.LabelColumn)
                .Append(mlContext.Transforms.LoadRawImageBytes(
                    outputColumnName: "ImageBytes", imageFolder: null, inputColumnName: "ImagePath"))
                .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(
                    featureColumnName: "ImageBytes", labelColumnName: "Label"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var trialChannel = progress is null ? null : new TrialProgressChannel(progress);

            var model = pipeline.Fit(trainSet);
            var predictions = model.Transform(testSet);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label");

            trialChannel?.ReportCompleted("ImageClassification (TF)", "accuracy", metrics.MacroAccuracy);

            return new AutoMLResult
            {
                BestTrainer = "ImageClassification (TensorFlow)",
                Model = model,
                Metrics = new Dictionary<string, double>
                {
                    ["accuracy"] = metrics.MacroAccuracy,
                    ["micro_accuracy"] = metrics.MicroAccuracy,
                    ["log_loss"] = metrics.LogLoss
                },
                RowCount = trainSet.GetRowCount() ?? 0
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    public static async Task<AutoMLResult> RunTextClassificationAsync(
        MLContext mlContext, Action<string> log,
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            var textCol = TextColumnFinder.FindFirst(trainSet.Schema, config.LabelColumn)
                ?? throw new InvalidOperationException("No text column found for text classification.");

            log($"Text classification: text='{textCol}', label='{config.LabelColumn}'");

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", config.LabelColumn)
                .Append(mlContext.MulticlassClassification.Trainers.TextClassification(
                    labelColumnName: "Label", sentence1ColumnName: textCol))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var trialChannel = progress is null ? null : new TrialProgressChannel(progress);

            var model = pipeline.Fit(trainSet);
            var predictions = model.Transform(testSet);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label");

            trialChannel?.ReportCompleted("TextClassification (NAS-BERT)", "accuracy", metrics.MacroAccuracy);

            return new AutoMLResult
            {
                BestTrainer = "TextClassification (NAS-BERT)",
                Model = model,
                Metrics = new Dictionary<string, double>
                {
                    ["accuracy"] = metrics.MacroAccuracy,
                    ["micro_accuracy"] = metrics.MicroAccuracy,
                    ["log_loss"] = metrics.LogLoss
                },
                RowCount = trainSet.GetRowCount() ?? 0
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    public static async Task<AutoMLResult> RunSentenceSimilarityAsync(
        MLContext mlContext, Action<string> log,
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            var textCols = TextColumnFinder.Find(trainSet.Schema, config.LabelColumn, 2);
            if (textCols.Count < 2)
                throw new InvalidOperationException("Sentence similarity requires at least two text columns.");

            log($"Sentence similarity: s1='{textCols[0]}', s2='{textCols[1]}', label='{config.LabelColumn}'");

            var pipeline = mlContext.Regression.Trainers.SentenceSimilarity(
                labelColumnName: config.LabelColumn,
                sentence1ColumnName: textCols[0],
                sentence2ColumnName: textCols[1]);

            var trialChannel = progress is null ? null : new TrialProgressChannel(progress);

            var model = pipeline.Fit(trainSet);
            var predictions = model.Transform(testSet);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: config.LabelColumn);

            trialChannel?.ReportCompleted("SentenceSimilarity (NAS-BERT)", "r_squared", metrics.RSquared);

            return new AutoMLResult
            {
                BestTrainer = "SentenceSimilarity (NAS-BERT)",
                Model = model,
                Metrics = new Dictionary<string, double>
                {
                    ["r_squared"] = metrics.RSquared,
                    ["rmse"] = metrics.RootMeanSquaredError,
                    ["mae"] = metrics.MeanAbsoluteError
                },
                RowCount = trainSet.GetRowCount() ?? 0
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    public static async Task<AutoMLResult> RunNerAsync(
        MLContext mlContext, Action<string> log,
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            var textCol = TextColumnFinder.FindFirst(trainSet.Schema, config.LabelColumn)
                ?? throw new InvalidOperationException("No text column found for NER.");

            log($"NER: text='{textCol}', label='{config.LabelColumn}'");

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", config.LabelColumn)
                .Append(mlContext.MulticlassClassification.Trainers.NamedEntityRecognition(
                    labelColumnName: "Label", sentence1ColumnName: textCol))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var trialChannel = progress is null ? null : new TrialProgressChannel(progress);

            var model = pipeline.Fit(trainSet);
            var predictions = model.Transform(testSet);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label");

            trialChannel?.ReportCompleted("NER (NAS-BERT)", "accuracy", metrics.MacroAccuracy);

            return new AutoMLResult
            {
                BestTrainer = "NER (NAS-BERT)",
                Model = model,
                Metrics = new Dictionary<string, double>
                {
                    ["accuracy"] = metrics.MacroAccuracy,
                    ["micro_accuracy"] = metrics.MicroAccuracy
                },
                RowCount = trainSet.GetRowCount() ?? 0
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    public static async Task<AutoMLResult> RunObjectDetectionAsync(
        MLContext mlContext, Action<string> log,
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            var labelColumn = string.IsNullOrWhiteSpace(config.LabelColumn)
                ? CocoDataLoader.DefaultLabelColumn
                : config.LabelColumn;

            log($"Object detection: label='{labelColumn}', AutoFormerV2 transfer learning");

            // CocoDataLoader produces three columns: ImagePath (string), the label vector
            // (VBuffer<string>, one class name per object), and BoundingBoxes (VBuffer<float>,
            // four values per object in x0 y0 x1 y1 order). The AutoFormerV2 ObjectDetection
            // trainer requires the image as an MLImage, the label as a vector of keys, and the
            // bounding-box float vector as-is — so LoadImages converts the path and
            // MapValueToKey converts the label vector before fitting.
            var pipeline = mlContext.Transforms.LoadImages(
                    outputColumnName: "Image", imageFolder: string.Empty, inputColumnName: CocoDataLoader.ImagePathColumn)
                .Append(mlContext.Transforms.Conversion.MapValueToKey(
                    outputColumnName: "LabelKey", inputColumnName: labelColumn))
                .Append(mlContext.MulticlassClassification.Trainers.ObjectDetection(
                    labelColumnName: "LabelKey",
                    boundingBoxColumnName: CocoDataLoader.BoundingBoxColumn,
                    imageColumnName: "Image"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(
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

            // No trial is reported: this handler computes no metrics (see the empty Metrics below),
            // and the progress channel carries a metric value by construction — the previous
            // report said accuracy=0, which for a detector reads as "found nothing".
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

    public static async Task<AutoMLResult> RunQuestionAnsweringAsync(
        MLContext mlContext, Action<string> log,
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            var textCols = TextColumnFinder.Find(trainSet.Schema, config.LabelColumn, 2);
            var contextCol = textCols.Count > 0 ? textCols[0] : throw new InvalidOperationException("No context column found.");
            var questionCol = textCols.Count > 1 ? textCols[1] : contextCol;

            log($"Question answering: context='{contextCol}', question='{questionCol}', answer='{config.LabelColumn}'");

            var pipeline = mlContext.MulticlassClassification.Trainers.QuestionAnswer(
                contextColumnName: contextCol,
                questionColumnName: questionCol);

            // No trial is reported — this handler computes no metrics, same as object detection above.
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
