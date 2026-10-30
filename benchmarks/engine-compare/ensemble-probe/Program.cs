using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Search;
using MLoop.Search.Candidates;

// Measures what an equal-weight average of the search's best distinct trials gains over its best single model - the lower bound of
// what a stacked ensemble could gain (a weight fitted on out-of-fold answers is no worse than equal weights). Outside the product:
// it runs the product's own search (TrialRunner) and refits the top settings, so the only difference between "single" and "average"
// is the average. args: train.csv test.csv label task(binary|multiclass|regression) budgetSeconds seed [maxK=5]
// Prints one JSON object: the best single model's test metric (binary: auc, multiclass: log_loss, regression: rmse), every member's, the
// average of the first K members for K = 2..maxK, and `forward` - forward selection with replacement on the members' out-of-fold answers
// (the member counts it chose, its out-of-fold metric against the best member's, and its test metric).
var (trainPath, testPath, label, task, budget, seed) = (args[0], args[1], args[2], args[3], int.Parse(args[4]), int.Parse(args[5]));
var maxK = args.Length > 6 ? int.Parse(args[6]) : 5;
var metricName = task switch { "binary" => "auc", "multiclass" => "log_loss", _ => "rmse" };
var higherIsBetter = task == "binary";

var ml = new MLContext(seed);
var (kinds, header) = Kinds(trainPath);
kinds[Array.IndexOf(header, label)] = false;   // the label is read as text and mapped to the task's form below
var loader = ml.Data.CreateTextLoader(new TextLoader.Options
{
    HasHeader = true,
    Separators = [','],
    AllowQuoting = true,
    MissingRealsAsNaNs = true,
    Columns = [.. header.Select((h, i) => new TextLoader.Column(h, kinds[i] ? DataKind.Single : DataKind.String, i))],
});
var train = loader.Load(trainPath);
var test = loader.Load(testPath);

// The label in the form the task's candidates read: bool, key or Single.
var classes = train.GetColumn<string>(label).Concat(test.GetColumn<string>(label)).Distinct().Order(StringComparer.Ordinal).ToList();
IEstimator<ITransformer> labeller = task switch
{
    "binary" => ml.Transforms.Conversion.MapValue(label, [new KeyValuePair<string, bool>(classes[0], false), new KeyValuePair<string, bool>(classes[1], true)], label),
    "multiclass" => ml.Transforms.Conversion.MapValueToKey(label, label, keyData: ml.Data.LoadFromEnumerable(classes.Select(c => new Cls { Value = c })), keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue),
    _ => ml.Transforms.Conversion.ConvertType(label, label, DataKind.Single),
};
if (task == "binary" && classes.Count != 2)
    throw new InvalidOperationException($"binary needs two classes, the table has {classes.Count}");
var labelling = labeller.Fit(train);
train = labelling.Transform(train);
test = labelling.Transform(test);

var roles = new ColumnRoles(label, header.Select((h, i) => new KeyValuePair<string, ColumnRole>(h, h == label ? ColumnRole.Ignored
    : kinds[i] ? ColumnRole.Numeric : Distinct(train, h) <= 100 ? ColumnRole.Categorical : ColumnRole.Text)).Where(c => c.Key != label));
var settings = new SearchSettings(Seed: seed);
var candidates = task switch { "binary" => CandidateCatalog.Binary(), "multiclass" => CandidateCatalog.Multiclass(), _ => CandidateCatalog.Regression() };
var policy = new Policy(metricName, higherIsBetter);

IReadOnlyDictionary<string, double> ScoreFold(IDataView scored) => new Dictionary<string, double>
{
    [metricName] = task switch
    {
        "binary" => ml.BinaryClassification.EvaluateNonCalibrated(scored, label).AreaUnderRocCurve,
        "multiclass" => ml.MulticlassClassification.Evaluate(scored, label).LogLoss,
        _ => ml.Regression.Evaluate(scored, label).RootMeanSquaredError,
    },
};

var clock = Stopwatch.StartNew();
var request = new SearchRequest(candidates, policy, ScoreFold, TimeSpan.FromSeconds(budget), settings, Stratify: task != "regression");
var search = new TrialRunner(request).Run(train, roles);
var searchSeconds = clock.Elapsed.TotalSeconds;

// The first maxK distinct (trainer, settings) of the trials that scored, best first.
var ranked = search.Trials.Where(t => t.Succeeded && t.Metrics!.TryGetValue(metricName, out var v) && double.IsFinite(v))
    .OrderBy(t => higherIsBetter ? -t.Metrics![metricName] : t.Metrics![metricName])
    .GroupBy(t => t.Trainer + "|" + string.Join(",", (t.Parameters ?? []).Select(p => p.Key + "=" + p.Value)))
    .Select(g => g.First()).Take(maxK).ToList();

var withMissing = Featurizer.ColumnsWithMissingValues(train, roles);
var columns = new TrainerColumns(label, "Features", null);
var members = new List<(TrialResult Trial, double Test, double[][] Scores, double Reference)>();
double[]? truth = null;
foreach (var trial in ranked)
{
    var candidate = candidates.First(c => c.Name == trial.Trainer);
    var values = (trial.Parameters ?? []).ToDictionary(p => p.Key, p => double.Parse(p.Value, CultureInfo.InvariantCulture));
    var point = candidate.Space.PointOf(values);
    var context = new MLContext(seed);
    var pipeline = Featurizer.Build(context, train.Schema, roles, withMissing, "Features")!.Append(candidate.Build(context, point, columns, settings));
    var predicted = pipeline.Fit(train).Transform(test);
    var scores = Scores(predicted, task);
    // The same metric from ML.NET's own evaluation of the predictions: a cross-check of the score-column order used below.
    // The answer is read from the predictions' own label column: the pipeline keys a class label itself, and the score slots follow its order.
    truth ??= Truth(predicted, label, task);
    members.Add((trial, Metric(task, scores, truth), scores, ScoreFold(predicted)[metricName]));
}

double AverageMetric(int k) => Metric(task, Average(members.Take(k).Select(m => m.Scores).ToList()), truth!);

// Forward selection (Caruana et al. 2004): from the members' out-of-fold answers on the TRAIN rows, greedily add - with replacement - the member
// whose addition most improves the running average's metric, and keep the best prefix. The weights are the member counts of that prefix and are
// chosen without the test rows, which only score the result. The out-of-fold answers come from a five-fold split of the train rows, a plain one
// (ML.NET has no stratified split) made once and shared by every member. A failure here drops only this arm: the equal-weight rows stay.
object? forward = null;
string? forwardError = null;
if (members.Count > 1)
{
    try
    {
        const int Steps = 20;
        var folds = ml.Data.CrossValidationSplit(train, numberOfFolds: 5, seed: seed);
        var oof = new List<double[][]>();
        var oofTruth = new List<double>();
        foreach (var member in members)
        {
            var candidate = candidates.First(c => c.Name == member.Trial.Trainer);
            var point = candidate.Space.PointOf((member.Trial.Parameters ?? []).ToDictionary(p => p.Key, p => double.Parse(p.Value, CultureInfo.InvariantCulture)));
            var rows = new List<double[]>();
            foreach (var fold in folds)
            {
                var context = new MLContext(seed);
                var pipeline = Featurizer.Build(context, fold.TrainSet.Schema, roles, withMissing, "Features")!.Append(candidate.Build(context, point, columns, settings));
                var predicted = pipeline.Fit(fold.TrainSet).Transform(fold.TestSet);
                rows.AddRange(Scores(predicted, task));
                if (oof.Count == 0) oofTruth.AddRange(Truth(predicted, label, task));
            }
            oof.Add([.. rows]);
        }
        var oofTruthArray = oofTruth.ToArray();
        bool Better(double a, double b) => higherIsBetter ? a > b : a < b;

        var counts = new int[members.Count];
        double[][]? sum = null;
        var total = 0;
        var bestScore = double.NaN;
        var bestCounts = (int[])counts.Clone();
        for (var step = 0; step < Steps; step++)
        {
            var (pick, pickScore) = (-1, double.NaN);
            for (var j = 0; j < members.Count; j++)
            {
                var score = Metric(task, Blend(sum, oof[j], total), oofTruthArray);
                if (double.IsFinite(score) && (pick < 0 || Better(score, pickScore))) (pick, pickScore) = (j, score);
            }
            if (pick < 0) break;
            sum = Add(sum, oof[pick]);
            counts[pick]++;
            total++;
            if (double.IsNaN(bestScore) || Better(pickScore, bestScore)) (bestScore, bestCounts) = (pickScore, (int[])counts.Clone());
        }
        var weighted = Weighted(members.Select(m => m.Scores).ToList(), bestCounts);
        var oofSingles = oof.Select(o => Metric(task, o, oofTruthArray)).ToArray();
        forward = new
        {
            steps = bestCounts.Sum(),
            counts = bestCounts,
            oof_selected = bestScore,
            oof_best_single = higherIsBetter ? oofSingles.Max() : oofSingles.Min(),
            test = Metric(task, weighted, truth!),
        };
    }
    catch (Exception e)
    {
        forwardError = e.GetType().Name + ": " + e.Message;
    }
}

var result = new
{
    task,
    metric = metricName,
    seed,
    budget,
    trials = search.Trials.Count,
    searchSeconds,
    best_single = members.Count > 0 ? members[0].Test : double.NaN,
    members = members.Select(m => new { m.Trial.Trainer, validation = m.Trial.Metrics![metricName], test = m.Test, reference = m.Reference }),
    average = Enumerable.Range(2, Math.Max(0, Math.Min(maxK, members.Count) - 1)).ToDictionary(k => "k" + k, AverageMetric),
    forward,
    forward_error = forwardError,
};
Console.WriteLine(JsonSerializer.Serialize(result, new JsonSerializerOptions { NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals }));
return 0;

static (bool[] Numeric, string[] Header) Kinds(string path)
{
    using var reader = new StreamReader(path);
    var header = Split(reader.ReadLine()!);
    var numeric = Enumerable.Repeat(true, header.Length).ToArray();
    for (var n = 0; n < 2000 && reader.ReadLine() is { } line; n++)
    {
        var cells = Split(line);
        for (var c = 0; c < header.Length && c < cells.Length; c++)
            if (numeric[c] && cells[c].Length > 0 && !double.TryParse(cells[c], NumberStyles.Float, CultureInfo.InvariantCulture, out _))
                numeric[c] = false;
    }
    return (numeric, header);
}

static string[] Split(string line)
{
    var cells = new List<string>();
    var cell = new System.Text.StringBuilder();
    var quoted = false;
    foreach (var ch in line)
    {
        if (ch == '"') quoted = !quoted;
        else if (ch == ',' && !quoted) { cells.Add(cell.ToString()); cell.Clear(); }
        else cell.Append(ch);
    }
    cells.Add(cell.ToString());
    return [.. cells];
}

static int Distinct(IDataView data, string column) => data.GetColumn<string>(column).Distinct().Count();

// Per row: binary [probability], multiclass [class probabilities], regression [prediction].
static double[][] Scores(IDataView predicted, string task)
{
    var schema = predicted.Schema;
    if (task == "multiclass")
        return [.. predicted.GetColumn<VBuffer<float>>("Score").Select(v => v.DenseValues().Select(x => (double)x).ToArray())];
    if (task == "regression")
        return [.. predicted.GetColumn<float>("Score").Select(s => new[] { (double)s })];
    // A binary trainer without a calibrated probability reports a raw score; a logistic of it puts every member on one scale.
    return schema.GetColumnOrNull("Probability") is not null
        ? [.. predicted.GetColumn<float>("Probability").Select(p => new[] { (double)p })]
        : [.. predicted.GetColumn<float>("Score").Select(s => new[] { 1 / (1 + Math.Exp(-s)) })];
}

static double[][] Average(IReadOnlyList<double[][]> members)
    => [.. Enumerable.Range(0, members[0].Length).Select(r => Enumerable.Range(0, members[0][r].Length).Select(c => members.Average(m => m[r][c])).ToArray())];

// (sum + add) / (n + 1) per cell: the average of n members already summed, with one more.
static double[][] Blend(double[][]? sum, double[][] add, int n)
    => [.. add.Select((row, r) => row.Select((x, c) => ((sum is null ? 0 : sum[r][c]) + x) / (n + 1)).ToArray())];

static double[][] Add(double[][]? sum, double[][] add)
    => [.. add.Select((row, r) => row.Select((x, c) => (sum is null ? 0 : sum[r][c]) + x).ToArray())];

// The members' answers averaged with the given counts as weights.
static double[][] Weighted(IReadOnlyList<double[][]> members, int[] counts)
{
    var total = (double)counts.Sum();
    return [.. Enumerable.Range(0, members[0].Length).Select(r => Enumerable.Range(0, members[0][r].Length)
        .Select(c => Enumerable.Range(0, members.Count).Sum(j => counts[j] * members[j][r][c]) / total).ToArray())];
}

static double[] Truth(IDataView test, string label, string task) => task switch
{
    "binary" => [.. test.GetColumn<bool>(label).Select(b => b ? 1.0 : 0.0)],
    "multiclass" => [.. test.GetColumn<uint>(label).Select(k => (double)(k - 1))],
    _ => [.. test.GetColumn<float>(label).Select(v => (double)v)],
};

static double Metric(string task, double[][] scores, double[] truth)
{
    switch (task)
    {
        case "binary":
            return Auc(scores.Select(s => s[0]).ToArray(), truth);
        case "multiclass":
            return Enumerable.Range(0, truth.Length).Average(i =>
            {
                var row = scores[i];
                var sum = row.Sum();
                return -Math.Log(Math.Clamp(row[(int)truth[i]] / (sum > 0 ? sum : 1), 1e-15, 1));
            });
        default:
            return Math.Sqrt(Enumerable.Range(0, truth.Length).Average(i => Math.Pow(scores[i][0] - truth[i], 2)));
    }
}

// Area under the ROC curve by ranks, ties at the mean rank.
static double Auc(double[] score, double[] truth)
{
    var order = Enumerable.Range(0, score.Length).OrderBy(i => score[i]).ToArray();
    var rank = new double[score.Length];
    for (var i = 0; i < order.Length;)
    {
        var j = i;
        while (j + 1 < order.Length && score[order[j + 1]] == score[order[i]]) j++;
        for (var k = i; k <= j; k++) rank[order[k]] = (i + j) / 2.0 + 1;
        i = j + 1;
    }
    var positives = truth.Count(t => t > 0.5);
    var negatives = truth.Length - positives;
    if (positives == 0 || negatives == 0) return double.NaN;
    var rankSum = Enumerable.Range(0, truth.Length).Where(i => truth[i] > 0.5).Sum(i => rank[i]);
    return (rankSum - positives * (positives + 1) / 2.0) / ((double)positives * negatives);
}

internal sealed record Policy(string Name, bool HigherIsBetter) : IMetricPolicy;

internal sealed class Cls
{
    public string Value { get; set; } = "";
}
