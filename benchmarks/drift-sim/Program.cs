using System.Globalization;
using MLoop.Core.Monitoring;

// How `mloop drift`'s reading behaves on synthetic columns of several shapes: how often a window with nothing changed reads "moderate" or
// worse (the null rate), how often a window with one column moved by half a robust standard deviation reads that column so, and what a
// window with many columns each moved a little reads. It builds the profile with the product's `DriftProfile.Build` and reads windows
// with `DriftMeasure.Compare` - the levels printed are the product's own. Synthetic data only; nothing is read from disk.
//
// args (key=value): train=5000 window=300 columns=200 profiles=20 windows=50 seed=1 families=normal,correlated,...
//   train   rows the profile is built from        window  rows in each window       columns  columns in the table (all one family)
//   profiles  independent profiles (the profile is itself a sample: its error is systematic within one, so several are drawn)
//   windows   windows read per profile, for each of the three kinds below
// Window kinds: null (nothing moved) | one (column c0 moved by 0.5 of the family's robust sigma) | many (columns c0..c19 each moved by 0.2).
// A window's columns are independent draws within a profile, but the profile is one sample: its error repeats in every window read against it,
// so the null rates carry a standard error of about sqrt(rate * (1 - rate) / profiles) - use many profiles (profiles=200 windows=5) to resolve a rate.
// Prints a markdown table. A window "reads moderate" when any column's level is Moderate or Significant; for `one` it is c0's level.
var options = args.Select(a => a.Split('=', 2)).ToDictionary(p => p[0], p => p.Length > 1 ? p[1] : "");
int Opt(string key, int fallback) => options.TryGetValue(key, out var v) ? int.Parse(v, CultureInfo.InvariantCulture) : fallback;
var trainRows = Opt("train", 5000);
var windowRows = Opt("window", 300);
var columns = Opt("columns", 200);
var profiles = Opt("profiles", 20);
var windowsPerProfile = Opt("windows", 50);
var seed = Opt("seed", 1);
const int ManyMoved = 20;
const double ManyScale = 0.4; // 0.4 of the 0.5 sigma unit = 0.2 sigma

var families = new (string Name, Func<Random, int, IReadOnlySet<int>, double, string[]> Row)[]
{
    ("normal", (rng, c, moved, scale) => Row(c, j => Normal(rng) + Moved(moved, j, 0.5 * scale))),
    ("correlated", CorrelatedRow),
    ("heavy-tail t2", (rng, c, moved, scale) => Row(c, j => Normal(rng) / Math.Sqrt(-Math.Log(1 - rng.NextDouble())) + Moved(moved, j, 0.5 * 1.21 * scale))),
    ("lognormal 1.5", (rng, c, moved, scale) => Row(c, j => Math.Exp(1.5 * Normal(rng)) + Moved(moved, j, 0.5 * 1.77 * scale))),
    ("poisson 2", (rng, c, moved, scale) => RowInt(c, j => Poisson(rng, 2 + (moved.Contains(j) ? 0.71 * scale : 0)))),
    ("integer 0-49", (rng, c, moved, scale) => RowInt(c, j => rng.Next(50) + (moved.Contains(j) ? (int)Math.Round(7.2 * scale) : 0))),
    // 60 % zeros, the rest exponential. The "moved" unit is the zero share: 10 points down (population PSI about 0.05, under the usual 0.1 on purpose:
    // a half-sigma has no meaning here) and 25 points down (population PSI about 0.26).
    ("zero-inflated", (rng, c, moved, scale) => Row(c, j => rng.NextDouble() < 0.6 - (moved.Contains(j) ? 0.1 * scale : 0) ? 0 : -Math.Log(1 - rng.NextDouble()))),
    ("zero-inflated big", (rng, c, moved, scale) => Row(c, j => rng.NextDouble() < 0.6 - (moved.Contains(j) ? 0.25 * scale : 0) ? 0 : -Math.Log(1 - rng.NextDouble()))),
};
var chosen = options.TryGetValue("families", out var list) ? list.Split(',') : null;
var schema = Enumerable.Range(0, columns).Select(j => ($"c{j}", true)).ToList();
IReadOnlySet<int> none = new HashSet<int>();
IReadOnlySet<int> first = new HashSet<int> { 0 };
IReadOnlySet<int> many = Enumerable.Range(0, ManyMoved).ToHashSet();

Console.WriteLine($"train {trainRows} rows, window {windowRows} rows, {columns} columns, {profiles} profiles x {windowsPerProfile} windows per kind, seed {seed}");
Console.WriteLine("| family | null: moderate+ | null: significant | one column 0.5 sigma: that column moderate+ | one column: significant | 20 columns 0.2 sigma: any moderate+ | many: a moved column moderate+ |");
Console.WriteLine("|---|---|---|---|---|---|---|");
foreach (var (name, row) in families)
{
    if (chosen is not null && !chosen.Contains(name.Split(' ')[0]) && !chosen.Contains(name))
        continue;
    int nullModerate = 0, nullSignificant = 0, oneModerate = 0, oneSignificant = 0, manyAny = 0, manyMoved = 0, total = 0;
    for (var p = 0; p < profiles; p++)
    {
        var rng = new Random(unchecked(seed * 100_003 + p));
        var train = Enumerable.Range(0, trainRows).Select(_ => row(rng, columns, none, 0)).ToList();
        var profile = DriftProfile.Build(train, schema);
        for (var w = 0; w < windowsPerProfile; w++)
        {
            total++;
            var quiet = DriftMeasure.Compare(profile, Window(row, rng, none, 0));
            nullModerate += quiet.Level >= DriftLevel.Moderate ? 1 : 0;
            nullSignificant += quiet.Level >= DriftLevel.Significant ? 1 : 0;
            var one = DriftMeasure.Compare(profile, Window(row, rng, first, 1)).Features.First(f => f.Column == "c0");
            oneModerate += one.Level >= DriftLevel.Moderate ? 1 : 0;
            oneSignificant += one.Level >= DriftLevel.Significant ? 1 : 0;
            var spread = DriftMeasure.Compare(profile, Window(row, rng, many, ManyScale));
            manyAny += spread.Level >= DriftLevel.Moderate ? 1 : 0;
            manyMoved += spread.Features.Any(f => f.Level >= DriftLevel.Moderate && int.Parse(f.Column[1..], CultureInfo.InvariantCulture) < ManyMoved) ? 1 : 0;
        }
    }
    string R(int n) => (n / (double)total).ToString("0.000", CultureInfo.InvariantCulture);
    Console.WriteLine($"| {name} | {R(nullModerate)} | {R(nullSignificant)} | {R(oneModerate)} | {R(oneSignificant)} | {R(manyAny)} | {R(manyMoved)} |");
}
return 0;

List<IReadOnlyDictionary<string, object?>> Window(Func<Random, int, IReadOnlySet<int>, double, string[]> row, Random rng, IReadOnlySet<int> moved, double scale)
    => [.. Enumerable.Range(0, windowRows).Select(_ =>
    {
        var cells = row(rng, columns, moved, scale);
        return (IReadOnlyDictionary<string, object?>)Enumerable.Range(0, columns).ToDictionary(j => $"c{j}", j => (object?)cells[j]);
    })];

static double Moved(IReadOnlySet<int> moved, int column, double amount) => moved.Contains(column) ? amount : 0;

static string[] Row(int columns, Func<int, double> draw)
    => [.. Enumerable.Range(0, columns).Select(j => draw(j).ToString("R", CultureInfo.InvariantCulture))];

static string[] RowInt(int columns, Func<int, int> draw)
    => [.. Enumerable.Range(0, columns).Select(j => draw(j).ToString(CultureInfo.InvariantCulture))];

static double Normal(Random rng) => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());

static int Poisson(Random rng, double lambda)
{
    var limit = Math.Exp(-lambda);
    var k = 0;
    for (var product = rng.NextDouble(); product > limit; product *= rng.NextDouble())
        k++;
    return k;
}

// Ten-column blocks that share one factor (correlation 0.64 inside a block, 0 across), unit variance.
static string[] CorrelatedRow(Random rng, int columns, IReadOnlySet<int> moved, double scale)
{
    var factors = new Dictionary<int, double>();
    return Row(columns, j =>
    {
        var block = j / 10;
        if (!factors.TryGetValue(block, out var factor))
            factors[block] = factor = Normal(rng);
        return 0.8 * factor + 0.6 * Normal(rng) + Moved(moved, j, 0.5 * scale);
    });
}
