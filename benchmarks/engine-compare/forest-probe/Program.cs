using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

// Trains ML.NET's FastForest (a random forest) with one explicit configuration and prints the held-out score:
// ROC AUC for a binary label, RMSE for a regression label.
// args: train.csv test.csv label binary|regression config.json — every column numeric, a header row, a binary label as 0/1.
// config: trees, leaves, minLeaf, bagFraction, treeFeatures, splitFeatures, bins
var (trainPath, testPath, label, task, configJson) = (args[0], args[1], args[2], args[3], args[4]);
var config = JsonSerializer.Deserialize<Dictionary<string, double>>(configJson)!;

var ml = new MLContext(seed: 0);
var header = File.ReadLines(trainPath).First().Split(',');
var columns = header.Select((name, i) => new TextLoader.Column(name, name == label && task == "binary" ? DataKind.Boolean : DataKind.Single, i)).ToArray();
var loader = ml.Data.CreateTextLoader(new TextLoader.Options { Columns = columns, HasHeader = true, Separators = new[] { ',' }, MissingRealsAsNaNs = true });
var train = loader.Load(trainPath);
var test = loader.Load(testPath);
var features = header.Where(h => h != label).ToArray();
var concat = ml.Transforms.Concatenate("Features", features);

if (task == "binary")
{
    var options = new FastForestBinaryTrainer.Options
    {
        LabelColumnName = label,
        FeatureColumnName = "Features",
        NumberOfTrees = (int)config["trees"],
        NumberOfLeaves = (int)config["leaves"],
        MinimumExampleCountPerLeaf = (int)config["minLeaf"],
        BaggingExampleFraction = config["bagFraction"],
        FeatureFraction = config["treeFeatures"],
        FeatureFractionPerSplit = config["splitFeatures"],
        MaximumBinCountPerFeature = (int)config["bins"],
        Seed = 1,
    };
    var model = concat.Append(ml.BinaryClassification.Trainers.FastForest(options)).Fit(train);
    var metrics = ml.BinaryClassification.EvaluateNonCalibrated(model.Transform(test), labelColumnName: label);
    Console.WriteLine(JsonSerializer.Serialize(new { auc = metrics.AreaUnderRocCurve }));
}
else
{
    var options = new FastForestRegressionTrainer.Options
    {
        LabelColumnName = label,
        FeatureColumnName = "Features",
        NumberOfTrees = (int)config["trees"],
        NumberOfLeaves = (int)config["leaves"],
        MinimumExampleCountPerLeaf = (int)config["minLeaf"],
        BaggingExampleFraction = config["bagFraction"],
        FeatureFraction = config["treeFeatures"],
        FeatureFractionPerSplit = config["splitFeatures"],
        MaximumBinCountPerFeature = (int)config["bins"],
        Seed = 1,
    };
    var model = concat.Append(ml.Regression.Trainers.FastForest(options)).Fit(train);
    var metrics = ml.Regression.Evaluate(model.Transform(test), labelColumnName: label);
    Console.WriteLine(JsonSerializer.Serialize(new { rmse = metrics.RootMeanSquaredError }));
}
