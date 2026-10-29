using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

// Trains ML.NET's LightGBM regressor with one explicit configuration and prints the test RMSE.
// args: train.csv test.csv label config.json — every column numeric, a header row.
var (trainPath, testPath, label, configJson) = (args[0], args[1], args[2], args[3]);
var config = JsonSerializer.Deserialize<Dictionary<string, double>>(configJson)!;

var ml = new MLContext(seed: 0);
var header = File.ReadLines(trainPath).First().Split(',');
var columns = header.Select((name, i) => new TextLoader.Column(name, DataKind.Single, i)).ToArray();
var loader = ml.Data.CreateTextLoader(new TextLoader.Options { Columns = columns, HasHeader = true, Separators = new[] { ',' }, MissingRealsAsNaNs = true });
var train = loader.Load(trainPath);
var test = loader.Load(testPath);

var features = header.Where(h => h != label).ToArray();
var options = new LightGbmRegressionTrainer.Options
{
    LabelColumnName = label,
    FeatureColumnName = "Features",
    NumberOfLeaves = (int)config["leaves"],
    NumberOfIterations = (int)config["trees"],
    MinimumExampleCountPerLeaf = (int)config["minChild"],
    LearningRate = config["lr"],
    MaximumBinCountPerFeature = (int)config["bins"],
    Booster = new GradientBooster.Options
    {
        FeatureFraction = config["colsample"],
        L1Regularization = config["l1"],
        L2Regularization = config["l2"],
    },
    Seed = 0,
    Deterministic = true,
};
var pipeline = ml.Transforms.Concatenate("Features", features).Append(ml.Regression.Trainers.LightGbm(options));
var model = pipeline.Fit(train);
var metrics = ml.Regression.Evaluate(model.Transform(test), labelColumnName: label);

// What the model costs to keep: its file, the time to write and read it back, the time to score the held-out rows.
var path = Path.Combine(Path.GetTempPath(), $"mlnet-lgbm-{Guid.NewGuid():N}.zip");
var clock = System.Diagnostics.Stopwatch.StartNew();
ml.Model.Save(model, train.Schema, path);
var saveSeconds = clock.Elapsed.TotalSeconds;
var modelBytes = new FileInfo(path).Length;
clock.Restart();
var loaded = ml.Model.Load(path, out _);
var loadSeconds = clock.Elapsed.TotalSeconds;
clock.Restart();
var scored = loaded.Transform(test).GetColumn<float>("Score").Count();
var scoreSeconds = clock.Elapsed.TotalSeconds;
File.Delete(path);
Console.WriteLine(JsonSerializer.Serialize(new { rmse = metrics.RootMeanSquaredError, modelBytes, saveSeconds, loadSeconds, scoreSeconds, scoredRows = scored }));
