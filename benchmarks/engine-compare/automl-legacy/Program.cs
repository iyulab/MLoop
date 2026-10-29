using System.Diagnostics;
using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

// Runs ML.NET's own AutoML (Microsoft.ML.AutoML) on one train file for a time budget and scores the best model on the test file.
// args: train.csv test.csv label task(binary|multiclass|regression) budgetSeconds
// Prints one JSON object: the trainer it picked, how many runs it made, the wall time, and the metric the table reports
// (binary: auc, multiclass: log_loss, regression: rmse) — the same metric the search was told to optimize.
var (trainPath, testPath, label, task, budget) = (args[0], args[1], args[2], args[3], uint.Parse(args[4]));

var ml = new MLContext(seed: 0);
var inferred = ml.Auto().InferColumns(trainPath, label);
var loader = ml.Data.CreateTextLoader(inferred.TextLoaderOptions);
var train = loader.Load(trainPath);
var test = loader.Load(testPath);

// A binary experiment wants a Boolean label; a table whose classes are words (good / bad) is read as text, so the classes become false / true
// in sorted order — the same positive class the other systems score (ROC AUC does not depend on which class is called positive for a good model).
if (task == "binary" && train.Schema[label].Type is Microsoft.ML.Data.TextDataViewType)
{
    var classes = train.GetColumn<string>(label).Concat(test.GetColumn<string>(label)).Distinct().Order(StringComparer.Ordinal).ToList();
    if (classes.Count == 2)
    {
        var map = new[] { new KeyValuePair<string, bool>(classes[0], false), new KeyValuePair<string, bool>(classes[1], true) };
        var mapper = ml.Transforms.Conversion.MapValue(label, map, label).Fit(train);
        train = mapper.Transform(train);
        test = mapper.Transform(test);
    }
}

var clock = Stopwatch.StartNew();
string trainer;
int runs;
object metrics;
switch (task)
{
    case "binary":
        {
            var result = ml.Auto().CreateBinaryClassificationExperiment(new BinaryExperimentSettings
            {
                MaxExperimentTimeInSeconds = budget,
                OptimizingMetric = BinaryClassificationMetric.AreaUnderRocCurve,
            }).Execute(train, inferred.ColumnInformation);
            trainer = result.BestRun.TrainerName;
            runs = result.RunDetails.Count();
            metrics = new { auc = ml.BinaryClassification.EvaluateNonCalibrated(result.BestRun.Model.Transform(test), label).AreaUnderRocCurve };
            break;
        }
    case "multiclass":
        {
            var result = ml.Auto().CreateMulticlassClassificationExperiment(new MulticlassExperimentSettings
            {
                MaxExperimentTimeInSeconds = budget,
                OptimizingMetric = MulticlassClassificationMetric.LogLoss,
            }).Execute(train, inferred.ColumnInformation);
            trainer = result.BestRun.TrainerName;
            runs = result.RunDetails.Count();
            metrics = new { log_loss = ml.MulticlassClassification.Evaluate(result.BestRun.Model.Transform(test), label).LogLoss };
            break;
        }
    default:
        {
            var result = ml.Auto().CreateRegressionExperiment(new RegressionExperimentSettings
            {
                MaxExperimentTimeInSeconds = budget,
                OptimizingMetric = RegressionMetric.RootMeanSquaredError,
            }).Execute(train, inferred.ColumnInformation);
            trainer = result.BestRun.TrainerName;
            runs = result.RunDetails.Count();
            metrics = new { rmse = ml.Regression.Evaluate(result.BestRun.Model.Transform(test), label).RootMeanSquaredError };
            break;
        }
}
Console.WriteLine(JsonSerializer.Serialize(new { learner = trainer, trials = runs, wall_s = Math.Round(clock.Elapsed.TotalSeconds, 1), test = metrics }));
