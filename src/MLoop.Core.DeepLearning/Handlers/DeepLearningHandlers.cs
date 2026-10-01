using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using MLoop.Core.AutoML;
using MLoop.Core.Data;
using MLoop.Core.Evaluation;
using MLoop.Core.Models;

namespace MLoop.Core.DeepLearning;

/// <summary>
/// Deep-learning task handlers (TensorFlow/TorchSharp-backed): image-classification,
/// text-classification, sentence-similarity, ner, object-detection, question-answering.
///
/// Moved out of <c>MLoop.Core.AutoML.AutoMLRunner</c> into this
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

            var trialChannel = new TrialProgressChannel(progress);

            var model = pipeline.Fit(trainSet);
            var predictions = model.Transform(testSet);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label");

            var metricsDict = new Dictionary<string, double>
            {
                ["macro_accuracy"] = metrics.MacroAccuracy,
                ["micro_accuracy"] = metrics.MicroAccuracy,
                ["log_loss"] = metrics.LogLoss
            };

            trialChannel.ReportCompleted(
                TrainerDescriptor.Of("ImageClassification (TF)"), "accuracy", metrics.MacroAccuracy, metricsDict);

            return new AutoMLResult
            {
                Trainer = TrainerDescriptor.Of("ImageClassification (TensorFlow)"),
                Model = model,
                Metrics = metricsDict,
                RowCount = trainSet.GetRowCount() ?? 0,
                Trials = trialChannel.Records,
                RankingMetric = trialChannel.RankingMetric
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
            var textCol = TextColumnFinder.FindFirst(trainSet, config.LabelColumn, config.ColumnOverrides, log)
                ?? throw new InvalidOperationException("No text column found for text classification.");

            log($"Text classification: text='{textCol}', label='{config.LabelColumn}'");

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", config.LabelColumn)
                .Append(mlContext.MulticlassClassification.Trainers.TextClassification(
                    labelColumnName: "Label", sentence1ColumnName: textCol, maxEpochs: EpochProgress.MaxEpochs))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var trialChannel = new TrialProgressChannel(progress);

            ITransformer model;
            using (EpochProgress.Attach(mlContext, progress, "Text classification (NAS-BERT)", log))
                model = PretrainedWeights.Guard(() => pipeline.Fit(trainSet));
            var predictions = model.Transform(testSet);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label");

            var metricsDict = new Dictionary<string, double>
            {
                ["macro_accuracy"] = metrics.MacroAccuracy,
                ["micro_accuracy"] = metrics.MicroAccuracy,
                ["log_loss"] = metrics.LogLoss
            };

            trialChannel.ReportCompleted(
                TrainerDescriptor.Of("TextClassification (NAS-BERT)"), "accuracy", metrics.MacroAccuracy, metricsDict);

            return new AutoMLResult
            {
                Trainer = TrainerDescriptor.Of("TextClassification (NAS-BERT)"),
                Model = model,
                Metrics = metricsDict,
                RowCount = trainSet.GetRowCount() ?? 0,
                Trials = trialChannel.Records,
                RankingMetric = trialChannel.RankingMetric
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
            var textCols = TextColumnFinder.Find(trainSet, config.LabelColumn, 2, config.ColumnOverrides, log);
            if (textCols.Count < 2)
                throw new InvalidOperationException("Sentence similarity requires at least two text columns.");

            log($"Sentence similarity: s1='{textCols[0]}', s2='{textCols[1]}', label='{config.LabelColumn}'");

            var pipeline = mlContext.Regression.Trainers.SentenceSimilarity(
                labelColumnName: config.LabelColumn,
                sentence1ColumnName: textCols[0],
                sentence2ColumnName: textCols[1],
                maxEpochs: EpochProgress.MaxEpochs);

            var trialChannel = new TrialProgressChannel(progress);

            ITransformer model;
            using (EpochProgress.Attach(mlContext, progress, "Sentence similarity (NAS-BERT)", log))
                model = PretrainedWeights.Guard(() => pipeline.Fit(trainSet));
            var predictions = model.Transform(testSet);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: config.LabelColumn);

            var metricsDict = new Dictionary<string, double>
            {
                ["r_squared"] = metrics.RSquared,
                ["rmse"] = metrics.RootMeanSquaredError,
                ["mae"] = metrics.MeanAbsoluteError
            };

            trialChannel.ReportCompleted(
                TrainerDescriptor.Of("SentenceSimilarity (NAS-BERT)"), "r_squared", metrics.RSquared, metricsDict);

            return new AutoMLResult
            {
                Trainer = TrainerDescriptor.Of("SentenceSimilarity (NAS-BERT)"),
                Model = model,
                Metrics = metricsDict,
                RowCount = trainSet.GetRowCount() ?? 0,
                Trials = trialChannel.Records,
                RankingMetric = trialChannel.RankingMetric
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
            var textCol = TextColumnFinder.FindFirst(trainSet, config.LabelColumn, config.ColumnOverrides, log)
                ?? throw new InvalidOperationException("No text column found for NER.");

            log($"NER: text='{textCol}', label='{config.LabelColumn}' (one tag per word, space-separated)");

            // The trainer wants a vector of tag keys per sentence; a CSV cell holds the tags as one
            // string. The split is fitted here and applied to the training and test data only — kept
            // out of the saved model, which would otherwise demand a label column at prediction.
            const string tags = "__NerTags";
            const string tagKeys = "__NerTagKeys";
            var labelPrep = mlContext.Transforms.Text.TokenizeIntoWords(tags, config.LabelColumn, [MLoop.Core.Prediction.TagSequence.Separator])
                .Append(mlContext.Transforms.Conversion.MapValueToKey(tagKeys, tags))
                .Fit(trainSet);
            var preparedTrain = labelPrep.Transform(trainSet);

            var pipeline = mlContext.MulticlassClassification.Trainers.NamedEntityRecognition(
                    labelColumnName: tagKeys, outputColumnName: "PredictedLabel",
                    sentence1ColumnName: textCol, maxEpochs: EpochProgress.MaxEpochs)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var trialChannel = new TrialProgressChannel(progress);

            ITransformer model;
            using (EpochProgress.Attach(mlContext, progress, "NER (NAS-BERT)", log))
                model = PretrainedWeights.Guard(() => pipeline.Fit(preparedTrain));

            var (micro, macro) = NerTagAccuracy.Measure(mlContext, model.Transform(testSet), config.LabelColumn);
            // Named as multiclass names them, and for the same reason: the tag average is what a model
            // that answers O everywhere cannot fake, and the gate's 1/N floor applies to it as-is.
            var metricsDict = new Dictionary<string, double>
            {
                ["macro_accuracy"] = macro,
                ["micro_accuracy"] = micro
            };

            trialChannel.ReportCompleted(TrainerDescriptor.Of("NER (NAS-BERT)"), "macro_accuracy", macro, metricsDict);

            return new AutoMLResult
            {
                Trainer = TrainerDescriptor.Of("NER (NAS-BERT)"),
                Model = model,
                Metrics = metricsDict,
                RowCount = trainSet.GetRowCount() ?? 0,
                Trials = trialChannel.Records,
                RankingMetric = trialChannel.RankingMetric
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
                    imageColumnName: "Image",
                    maxEpoch: EpochProgress.MaxEpochs))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                    outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel"));

            // real-data OD training intermittently dies with a native access violation
            // (0xC0000005) inside libtorch, crash location varying between runs (Linear_forward,
            // Tensor.backward). Upstream research (dotnet/TorchSharp#1292) traces this class of
            // crash to native heap corruption whose exact failure point shifts with memory/thread
            // pressure — forcing libtorch to a single thread removes that pressure source. This is
            // a defensive mitigation, not a confirmed fix (unverified under this investigation's
            // resource-constrained environment); it carries no downside beyond
            // slower CPU training, so it is applied unconditionally rather than gated on success.
            TorchSharp.torch.set_num_threads(1);

            // No trial is reported: this handler computes no metrics (see the empty Metrics below),
            // and the progress channel carries a metric value by construction — the previous
            // report said accuracy=0, which for a detector reads as "found nothing".
            ITransformer model;
            using (EpochProgress.Attach(mlContext, progress, "Object detection (AutoFormerV2)", log))
                model = PretrainedWeights.Guard(() => pipeline.Fit(trainSet));
            var predictions = model.Transform(testSet);

            return new AutoMLResult
            {
                Trainer = TrainerDescriptor.Of("ObjectDetection (AutoFormerV2)"),
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
            var answerCol = config.LabelColumn;
            var textCols = TextColumnFinder.Find(trainSet, answerCol, 2, config.ColumnOverrides, log);
            if (textCols.Count < 2)
                throw new InvalidOperationException(
                    "Question answering needs two text columns besides the answer — the passage and the question. " +
                    $"Found {textCols.Count}: {string.Join(", ", textCols)}.");

            // The passage is the longer of the two: which column is declared or reads more like language
            // does not say which one holds the answer, and a question used as the passage trains nothing.
            var (contextCol, questionCol) = MeanLength(trainSet, textCols[0]) >= MeanLength(trainSet, textCols[1])
                ? (textCols[0], textCols[1])
                : (textCols[1], textCols[0]);
            var startCol = AnswerStartColumn.Find(trainSet, contextCol, answerCol, [contextCol, questionCol, answerCol])
                ?? throw new InvalidOperationException(
                    $"Question answering needs the answer's start position in '{contextCol}': a column of whole " +
                    $"numbers where '{contextCol}' read from that position gives '{answerCol}'. No column did.");

            log($"Question answering: context='{contextCol}', question='{questionCol}', answer='{answerCol}', start='{startCol}'");

            // The answer and its start are what the trainer learns from; a prediction needs only the passage
            // and the question. So the training rows are prepared outside the model, as NER prepares its
            // tags: each start is located (past whitespace a trimmed answer no longer has) and made the
            // trainer's Int32, under the passage's and question's own names so the saved model reads them.
            var (prepared, dropped, writable, kept) = PrepareQuestionAnswerRows(mlContext, trainSet, contextCol, questionCol, answerCol, startCol);
            if (dropped > 0)
                log($"Question answering: {dropped} row(s) left out — the answer is not in the passage at the recorded position");
            if (QuestionAnswerAlphabet.Warning(writable, kept) is { } unwritable)
                log(unwritable);
            const string start = QuestionAnswerRow.StartColumn;

            var pipeline = mlContext.MulticlassClassification.Trainers.QuestionAnswer(
                contextColumnName: contextCol,
                questionColumnName: questionCol,
                trainingAnswerColumnName: answerCol,
                answerIndexColumnName: start,
                maxEpochs: EpochProgress.MaxEpochs);

            var trialChannel = new TrialProgressChannel(progress);

            ITransformer model;
            using (EpochProgress.Attach(mlContext, progress, "Question answering", log))
                model = PretrainedWeights.Guard(() => pipeline.Fit(prepared));

            var (exact, f1) = AnswerOverlap.Measure(model.Transform(testSet), answerCol);
            var metricsDict = new Dictionary<string, double>
            {
                [AnswerOverlap.CharF1] = f1,
                [AnswerOverlap.ExactMatch] = exact
            };

            trialChannel.ReportCompleted(TrainerDescriptor.Of("QA (RoBERTa)"), AnswerOverlap.CharF1, f1, metricsDict);

            return new AutoMLResult
            {
                Trainer = TrainerDescriptor.Of("QA (RoBERTa)"),
                Model = model,
                Metrics = metricsDict,
                RowCount = trainSet.GetRowCount() ?? 0,
                Trials = trialChannel.Records,
                RankingMetric = trialChannel.RankingMetric,
                TrainingOnlyColumns = [startCol]
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private static double MeanLength(IDataView data, string column)
    {
        var col = data.Schema[column];
        using var cursor = data.GetRowCursor([col]);
        var getter = cursor.GetGetter<ReadOnlyMemory<char>>(col);
        ReadOnlyMemory<char> value = default;
        long total = 0, rows = 0;
        while (rows < 500 && cursor.MoveNext())
        {
            getter(ref value);
            total += value.Length;
            rows++;
        }
        return rows == 0 ? 0 : (double)total / rows;
    }

    private sealed class QuestionAnswerRow
    {
        public const string StartColumn = "__AnswerStart";

        public string Context { get; set; } = "";
        public string Question { get; set; } = "";
        public string Answer { get; set; } = "";
        public int Start { get; set; }
    }

    private static (IDataView Rows, int Dropped, int Writable, int Kept) PrepareQuestionAnswerRows(
        MLContext mlContext, IDataView data, string contextCol, string questionCol, string answerCol, string startCol)
    {
        var columns = new[] { data.Schema[contextCol], data.Schema[questionCol], data.Schema[answerCol], data.Schema[startCol] };
        using var cursor = data.GetRowCursor(columns);
        var context = cursor.GetGetter<ReadOnlyMemory<char>>(columns[0]);
        var question = cursor.GetGetter<ReadOnlyMemory<char>>(columns[1]);
        var answer = cursor.GetGetter<ReadOnlyMemory<char>>(columns[2]);
        var position = AnswerStartColumn.Reader(cursor, columns[3]);

        var rows = new List<QuestionAnswerRow>();
        int dropped = 0, writable = 0;
        ReadOnlyMemory<char> c = default, q = default, a = default;
        while (cursor.MoveNext())
        {
            context(ref c);
            question(ref q);
            answer(ref a);
            if (AnswerStartColumn.Locate(c.Span, a.Span, position()) is { } at)
            {
                rows.Add(new QuestionAnswerRow { Context = c.ToString(), Question = q.ToString(), Answer = a.ToString(), Start = at });
                if (QuestionAnswerAlphabet.CanWrite(a.Span))
                    writable++;
            }
            else
                dropped++;
        }

        var schema = SchemaDefinition.Create(typeof(QuestionAnswerRow));
        schema[nameof(QuestionAnswerRow.Context)].ColumnName = contextCol;
        schema[nameof(QuestionAnswerRow.Question)].ColumnName = questionCol;
        schema[nameof(QuestionAnswerRow.Answer)].ColumnName = answerCol;
        schema[nameof(QuestionAnswerRow.Start)].ColumnName = QuestionAnswerRow.StartColumn;
        return (mlContext.Data.LoadFromEnumerable(rows, schema), dropped, writable, rows.Count);
    }
}
