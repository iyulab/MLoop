using System.Globalization;
using MLoop.Core.Monitoring;

// How `mloop drift`'s reading behaves on synthetic columns of several shapes: how often a window with nothing changed reads "moderate" or
// worse (the null rate), how often a window with one column moved by half a robust standard deviation reads that column so, and what a
// window with many columns each moved a little reads. It builds the profile with the product's `DriftProfile.Build` and reads windows
// with `DriftMeasure.Compare` - the levels printed are the product's own. Synthetic data only; nothing is read from disk.
//
// args (key=value): train=5000 window=300 columns=200 profiles=20 windows=50 seed=1 families=normal,correlated,...
//   train   rows the profile is built from        window  rows in each window       columns  columns in the table (all one family, or `mixed`: a family per column index mod 5)
//   profiles  independent profiles (the profile is itself a sample: its error is systematic within one, so several are drawn)
//   windows   windows read per profile, for each of the three kinds below
// Window kinds: null (nothing moved) | one (column c0 moved by 0.5 of the family's robust sigma, times onescale) | many (columns c0..c19 each moved by 0.2).
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
// onescale=0.6: how far the one moved column moves, in units of the half sigma (1 = 0.5 sigma, 0.6 = 0.3 sigma).
var oneScale = options.TryGetValue("onescale", out var oneScaleText) ? double.Parse(oneScaleText, CultureInfo.InvariantCulture) : 1.0;
// multipliers=2,4: also read each window as if the family error rate were divided by each (the bound is then worked out for that many times
// the columns), beside the reading as shipped. Needs only the public `NoDriftBound`, `LevelOf` and `OutsideRange`: the bound is rebuilt the way
// `DriftMeasure.Compare` builds it, including the two end bins a window that left the training range gets - multiplier 1 reproduces the
// shipped levels (the first line of a run with `multipliers=1,...` is that check).
var multipliers = options.TryGetValue("multipliers", out var multiplierList)
    ? multiplierList.Split(',').Select(m => double.Parse(m, CultureInfo.InvariantCulture)).ToArray()
    : [];
const int ManyMoved = 20;
const double ManyScale = 0.4; // 0.4 of the 0.5 sigma unit = 0.2 sigma

// Clustered rows: every 50 consecutive rows (training and windows alike) share a per-column effect, so rows inside a block are correlated (0.5)
// while each value is still N(0, 1) on its own - what a window of rows from one machine or one shift looks like.
var blockEffects = new double[columns];
var blockRows = 0;
// rho=0.5 block=50: the correlation inside a block and the rows per block (the defaults are a strong cluster; ordinary logs are weaker).
var blockCorrelation = options.TryGetValue("rho", out var rhoText) ? double.Parse(rhoText, CultureInfo.InvariantCulture) : 0.5;
var blockSize = Opt("block", 50);

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
    // Mixed table: the family of a column is its index mod 5 - 0 categorical (10 levels, level 0 holds 0.10; "moved" lifts it by 0.15 x scale),
    // 1 normal, 2 lognormal 1.5, 3 Poisson 2, 4 zero-inflated (60 % zeros). Column order is the schema's (see `MixedSchema`).
    ("mixed", (rng, c, moved, scale) => [.. Enumerable.Range(0, c).Select(j => (j % 5) switch
    {
        0 => (rng.NextDouble() < 0.10 + (moved.Contains(j) ? 0.15 * scale : 0) ? 0 : 1 + rng.Next(9)).ToString(CultureInfo.InvariantCulture),
        1 => (Normal(rng) + Moved(moved, j, 0.5 * scale)).ToString("R", CultureInfo.InvariantCulture),
        2 => (Math.Exp(1.5 * Normal(rng)) + Moved(moved, j, 0.5 * 1.77 * scale)).ToString("R", CultureInfo.InvariantCulture),
        3 => Poisson(rng, 2 + (moved.Contains(j) ? 0.71 * scale : 0)).ToString(CultureInfo.InvariantCulture),
        _ => (rng.NextDouble() < 0.6 - (moved.Contains(j) ? 0.1 * scale : 0) ? 0 : -Math.Log(1 - rng.NextDouble())).ToString("R", CultureInfo.InvariantCulture),
    })]),
    // Shape changes that keep the centre: the spread grows (scale), or a share of the values is drawn three times as far out (outliers). "moved" = the changed column;
    // the unmoved columns draw nothing extra, so their stream (and the null rate) is the normal family's.
    ("scale 1.5", (rng, c, moved, scale) => Row(c, j => Normal(rng) * (moved.Contains(j) ? 1 + 0.5 * scale : 1))),
    ("scale 1.25", (rng, c, moved, scale) => Row(c, j => Normal(rng) * (moved.Contains(j) ? 1 + 0.25 * scale : 1))),
    ("outliers 10%", (rng, c, moved, scale) => Row(c, j => Normal(rng) * (moved.Contains(j) && rng.NextDouble() < 0.10 * scale ? 3 : 1))),
    ("outliers 5%", (rng, c, moved, scale) => Row(c, j => Normal(rng) * (moved.Contains(j) && rng.NextDouble() < 0.05 * scale ? 3 : 1))),
    ("zero-inflated big", (rng, c, moved, scale) => Row(c, j => rng.NextDouble() < 0.6 - (moved.Contains(j) ? 0.25 * scale : 0) ? 0 : -Math.Log(1 - rng.NextDouble()))),
    // Categorical columns (names starting `cat_`; every column is categorical). "moved" is the changed column.
    // cat_new: 10 levels, the moved column draws a level training never saw ("new") for 5 % x scale of its values.
    ("cat_new", (rng, c, moved, scale) => RowText(c, j => moved.Contains(j) && rng.NextDouble() < 0.05 * scale ? "new" : rng.Next(10).ToString(CultureInfo.InvariantCulture))),
    // cat_rare: level 0 holds 0.5 % at training and 0.5 + 4.5 x scale % in the moved column; ten other levels share the rest.
    ("cat_rare", (rng, c, moved, scale) => RowText(c, j => rng.NextDouble() < 0.005 + (moved.Contains(j) ? 0.045 * scale : 0) ? "0" : (1 + rng.Next(10)).ToString(CultureInfo.InvariantCulture))),
    // cat_vanish: level 0 holds 10 % at training and 10 - 5 x scale % in the moved column; nine other levels share the rest.
    ("cat_vanish", (rng, c, moved, scale) => RowText(c, j => rng.NextDouble() < 0.10 - (moved.Contains(j) ? 0.05 * scale : 0) ? "0" : (1 + rng.Next(9)).ToString(CultureInfo.InvariantCulture))),
    // cat_highcard_new: 200 levels (the profile names the 20 commonest; the other 180 share its "other" bin), 5 % x scale of the moved column's values a new level.
    ("cat_highcard_new", (rng, c, moved, scale) => RowText(c, j => moved.Contains(j) && rng.NextDouble() < 0.05 * scale ? "new" : rng.Next(200).ToString(CultureInfo.InvariantCulture))),
    // A fifth of the cells are empty (the same in training and in windows).
    ("normal_missing20", (rng, c, moved, scale) => RowOrEmpty(rng, c, 0.2, j => Normal(rng) + Moved(moved, j, 0.5 * scale))),
    ("outliers10_missing20", (rng, c, moved, scale) => RowOrEmpty(rng, c, 0.2, j => Normal(rng) * (moved.Contains(j) && rng.NextDouble() < 0.10 * scale ? 3 : 1))),
    ("clustered", (rng, c, moved, scale) =>
    {
        if (blockRows++ % blockSize == 0)
            for (var j = 0; j < c; j++)
                blockEffects[j] = Normal(rng);
        return Row(c, j => Math.Sqrt(blockCorrelation) * blockEffects[j] + Math.Sqrt(1 - blockCorrelation) * Normal(rng) + Moved(moved, j, 0.5 * scale));
    }),
    ("clustered_outliers10", (rng, c, moved, scale) =>
    {
        if (blockRows++ % blockSize == 0)
            for (var j = 0; j < c; j++)
                blockEffects[j] = Normal(rng);
        return Row(c, j => (Math.Sqrt(blockCorrelation) * blockEffects[j] + Math.Sqrt(1 - blockCorrelation) * Normal(rng)) * (moved.Contains(j) && rng.NextDouble() < 0.10 * scale ? 3 : 1));
    }),
};
var chosen = options.TryGetValue("families", out var list) ? list.Split(',') : null;
var schema = Enumerable.Range(0, columns).Select(j => ($"c{j}", true)).ToList();
// In the mixed family every fifth column (0, 5, 10, ...) is categorical.
var mixedSchema = Enumerable.Range(0, columns).Select(j => ($"c{j}", j % 5 != 0)).ToList();
var categoricalSchema = Enumerable.Range(0, columns).Select(j => ($"c{j}", false)).ToList();
IReadOnlySet<int> none = new HashSet<int>();
IReadOnlySet<int> first = new HashSet<int> { 0 };
IReadOnlySet<int> many = Enumerable.Range(0, ManyMoved).ToHashSet();

// signal=range: instead of the PSI table, read the count of window values outside the training range (above the training maximum, below the
// training minimum) as a signal of its own - see `RangeSignal`.
options.TryGetValue("signal", out var signalMode);
if (signalMode == "range")
    return RangeSignal();

// signal=text: free-text columns read through their two descriptors (length, share of words outside the training vocabulary).
if (signalMode == "text")
    return TextSignal();

Console.WriteLine($"train {trainRows} rows, window {windowRows} rows, {columns} columns, {profiles} profiles x {windowsPerProfile} windows per kind, seed {seed}");
Console.WriteLine("| family | reading | null: moderate+ | null: significant | one column 0.5 sigma: that column moderate+ | one column: significant | 20 columns 0.2 sigma: any moderate+ | many: a moved column moderate+ |");
Console.WriteLine("|---|---|---|---|---|---|---|---|");
foreach (var (name, row) in families)
{
    if (chosen is not null && !chosen.Contains(name.Split(' ')[0]) && !chosen.Contains(name))
        continue;
    // Reading 0 is the one shipped; reading v (1-based) divides the family error rate by multipliers[v - 1].
    var readings = multipliers.Length + 1;
    var nullModerate = new int[readings];
    var nullSignificant = new int[readings];
    var oneModerate = new int[readings];
    var oneSignificant = new int[readings];
    var manyAny = new int[readings];
    var manyMoved = new int[readings];
    var total = 0;
    for (var p = 0; p < profiles; p++)
    {
        var rng = new Random(unchecked(seed * 100_003 + p));
        var train = Enumerable.Range(0, trainRows).Select(_ => row(rng, columns, none, 0)).ToList();
        var profile = DriftProfile.Build(train, name == "mixed" ? mixedSchema : name.StartsWith("cat_", StringComparison.Ordinal) ? categoricalSchema : schema);
        // The bound of each column under each alternative reading, for a window with nothing outside the training range; a column with stray
        // values gets the bound `Compare` builds for the extended bins.
        var plainBounds = multipliers
            .Select(m => profile.Features.Select(f => DriftMeasure.NoDriftBound(f.Shares, windowRows, profile.Rows, (int)Math.Round(profile.Features.Count * m))).ToArray())
            .ToArray();
        double[][] BoundsOf(List<IReadOnlyDictionary<string, object?>> window)
            => multipliers.Select((m, v) => profile.Features.Select((feature, col) =>
            {
                var (below, above) = feature.OutsideRange(window.Select(r => TextOf(r[feature.Column])).ToList());
                if (below + above == 0)
                    return plainBounds[v][col];
                var reference = feature.Shares;
                var withValue = 1 - reference[^1];
                var tail = withValue / (Math.Max(profile.Rows, 1) * withValue + 1);
                var expected = reference.Select(x => x * (1 - 2 * tail)).Append(tail).Append(tail).ToList();
                return DriftMeasure.NoDriftBound(expected, window.Count, profile.Rows, (int)Math.Round(profile.Features.Count * m));
            }).ToArray()).ToArray();
        // The level of one column's drift under reading v (0 = as shipped), for the window whose bounds are `windowBounds`.
        DriftLevel Level(FeatureDrift f, int v, double[][] windowBounds)
            => v == 0 ? f.Level : DriftMeasure.LevelOf(f.Psi, windowBounds[v - 1][int.Parse(f.Column[1..], CultureInfo.InvariantCulture)]);
        DriftLevel Worst(DriftReport report, int v, double[][] windowBounds)
            => report.Features.Count == 0 ? DriftLevel.None : report.Features.Max(f => Level(f, v, windowBounds));
        for (var w = 0; w < windowsPerProfile; w++)
        {
            total++;
            var quietWindow = Window(row, rng, none, 0);
            var oneWindow = Window(row, rng, first, oneScale);
            var spreadWindow = Window(row, rng, many, ManyScale);
            var quiet = DriftMeasure.Compare(profile, quietWindow);
            var one = DriftMeasure.Compare(profile, oneWindow).Features.First(f => f.Column == "c0");
            var spread = DriftMeasure.Compare(profile, spreadWindow);
            var quietBounds = BoundsOf(quietWindow);
            var oneBounds = BoundsOf(oneWindow);
            var spreadBounds = BoundsOf(spreadWindow);
            for (var v = 0; v < readings; v++)
            {
                nullModerate[v] += Worst(quiet, v, quietBounds) >= DriftLevel.Moderate ? 1 : 0;
                nullSignificant[v] += Worst(quiet, v, quietBounds) >= DriftLevel.Significant ? 1 : 0;
                oneModerate[v] += Level(one, v, oneBounds) >= DriftLevel.Moderate ? 1 : 0;
                oneSignificant[v] += Level(one, v, oneBounds) >= DriftLevel.Significant ? 1 : 0;
                manyAny[v] += Worst(spread, v, spreadBounds) >= DriftLevel.Moderate ? 1 : 0;
                manyMoved[v] += spread.Features.Any(f => Level(f, v, spreadBounds) >= DriftLevel.Moderate && int.Parse(f.Column[1..], CultureInfo.InvariantCulture) < ManyMoved) ? 1 : 0;
            }
        }
    }
    string R(int n) => (n / (double)total).ToString("0.000", CultureInfo.InvariantCulture);
    for (var v = 0; v < readings; v++)
    {
        var label = v == 0 ? "as shipped (0.05)" : $"error rate / {multipliers[v - 1].ToString(CultureInfo.InvariantCulture)} ({(0.05 / multipliers[v - 1]).ToString("0.####", CultureInfo.InvariantCulture)})";
        Console.WriteLine($"| {name} | {label} | {R(nullModerate[v])} | {R(nullSignificant[v])} | {R(oneModerate[v])} | {R(oneSignificant[v])} | {R(manyAny[v])} | {R(manyMoved[v])} |");
    }
}
return 0;

// How the count of window values outside the training range reads, apart from the PSI - the product's own `FeatureDrift.Range` (the exact
// chance of that many values beyond the training extreme, and the family-wise `Notable`). Prints: the share of windows with nothing moved where
// any column is notable (and where any column's p is under 1e-3, informational), and, for column c0 moved as the family says (`onescale`),
// how often it is notable beside how often the PSI reads that column moderate or worse as shipped.
int RangeSignal()
{
    Console.WriteLine($"range signal: train {trainRows} rows, window {windowRows} rows, {columns} columns, {profiles} profiles x {windowsPerProfile} windows per kind, seed {seed}");
    Console.WriteLine("| family | null: any column notable | null: any column p<1e-3 | c0 moved: range notable | c0 moved: p<1e-3 | c0 moved: PSI moderate+ as shipped | null: PSI moderate+ (any column) | null: PSI significant (what the retraining trigger reads) |");
    Console.WriteLine("|---|---|---|---|---|---|---|---|");
    foreach (var (name, row) in families)
    {
        if (name == "mixed" || (chosen is not null && !chosen.Contains(name.Split(' ')[0]) && !chosen.Contains(name)))
            continue;
        int nullNotable = 0, nullColumn = 0, oneNotable = 0, oneColumn = 0, onePsi = 0, nullModerate = 0, nullTrigger = 0, total = 0;
        for (var p = 0; p < profiles; p++)
        {
            var rng = new Random(unchecked(seed * 100_003 + p));
            var train = Enumerable.Range(0, trainRows).Select(_ => row(rng, columns, none, 0)).ToList();
            var profile = DriftProfile.Build(train, schema);
            for (var w = 0; w < windowsPerProfile; w++)
            {
                total++;
                var quiet = DriftMeasure.Compare(profile, Window(row, rng, none, 0));
                nullNotable += quiet.RangeNotable.Count > 0 ? 1 : 0;
                nullColumn += quiet.Features.Any(f => f.Range is { } r && r.P < 1e-3) ? 1 : 0;
                nullModerate += quiet.Level >= DriftLevel.Moderate ? 1 : 0;
                nullTrigger += quiet.Features.Any(f => f.Psi >= Math.Max(DriftMeasure.SignificantThreshold, f.Bound)) ? 1 : 0;
                var one = DriftMeasure.Compare(profile, Window(row, rng, first, oneScale)).Features.First(f => f.Column == "c0");
                oneNotable += one.Range is { Notable: true } ? 1 : 0;
                oneColumn += one.Range is { } r0 && r0.P < 1e-3 ? 1 : 0;
                onePsi += one.Level >= DriftLevel.Moderate ? 1 : 0;
            }
        }
        string R(int n) => (n / (double)total).ToString("0.000", CultureInfo.InvariantCulture);
        Console.WriteLine($"| {name} | {R(nullNotable)} | {R(nullColumn)} | {R(oneNotable)} | {R(oneColumn)} | {R(onePsi)} | {R(nullModerate)} | {R(nullTrigger)} |");
    }
    return 0;
}

// The sentences are Zipf-distributed words (vocabulary `vocab`, exponent 1) of 1 + Poisson(`words`) words; `fresh` is the chance that a word is
// replaced by one the training text never contained, `longer` multiplies the mean length. Prints, per shift, the share of windows where the
// column reads moderate or worse and where each descriptor does.
int TextSignal()
{
    var vocab = options.TryGetValue("vocab", out var v) ? int.Parse(v, CultureInfo.InvariantCulture) : 3000;
    var words = options.TryGetValue("words", out var wv) ? double.Parse(wv, CultureInfo.InvariantCulture) : 11.0;
    var cumulative = new double[vocab];
    var sum = 0.0;
    for (var i = 0; i < vocab; i++)
        cumulative[i] = sum += 1.0 / (i + 1);
    var fresh = 0;
    string Sentence(Random rng, double newShare, double longer)
    {
        var count = 1 + Poisson(rng, words * longer);
        var parts = new string[count];
        for (var k = 0; k < count; k++)
        {
            if (rng.NextDouble() < newShare)
            {
                parts[k] = "z" + (++fresh).ToString(CultureInfo.InvariantCulture);
                continue;
            }
            var target = rng.NextDouble() * sum;
            var lo = 0;
            var hi = vocab - 1;
            while (lo < hi)
            {
                var mid = (lo + hi) / 2;
                if (cumulative[mid] < target) lo = mid + 1; else hi = mid;
            }
            parts[k] = "w" + lo.ToString(CultureInfo.InvariantCulture);
        }
        return string.Join(' ', parts);
    }

    var shifts = new (string Name, double NewShare, double Longer)[]
    {
        ("nothing moved", 0, 1), ("5% new words", 0.05, 1), ("10% new words", 0.10, 1), ("20% new words", 0.20, 1), ("length x1.3", 0, 1.3),
    };
    Console.WriteLine($"text signal: train {trainRows} rows, window {windowRows} rows, vocabulary {vocab}, {words} words a sentence, {profiles} profiles x {windowsPerProfile} windows per shift, seed {seed}");
    Console.WriteLine("| shift | column moderate+ | column significant | length descriptor moderate+ | unseen-words descriptor moderate+ |");
    Console.WriteLine("|---|---|---|---|---|");
    var counts = new int[shifts.Length, 4];
    var total = 0;
    for (var p = 0; p < profiles; p++)
    {
        var rng = new Random(unchecked(seed * 100_003 + p));
        var train = Enumerable.Range(0, trainRows).Select(_ => new string?[] { Sentence(rng, 0, 1) }).ToList();
        var profile = DriftProfile.Build(train.Select(_ => Array.Empty<string?>()).ToList(), []) with
        {
            Texts = [TextProfile.Build("t0", train.Select(r => r[0]).ToList())],
        };
        total += windowsPerProfile;
        for (var w = 0; w < windowsPerProfile; w++)
            for (var s = 0; s < shifts.Length; s++)
            {
                var window = Enumerable.Range(0, windowRows)
                    .Select(_ => (IReadOnlyDictionary<string, object?>)new Dictionary<string, object?> { ["t0"] = Sentence(rng, shifts[s].NewShare, shifts[s].Longer) })
                    .ToList();
                var report = DriftMeasure.Compare(profile, window);
                counts[s, 0] += report.Features.Any(f => f.Level >= DriftLevel.Moderate) ? 1 : 0;
                counts[s, 1] += report.Features.Any(f => f.Level >= DriftLevel.Significant) ? 1 : 0;
                counts[s, 2] += report.Features.Any(f => f.Column == TextProfile.LengthColumn("t0") && f.Level >= DriftLevel.Moderate) ? 1 : 0;
                counts[s, 3] += report.Features.Any(f => f.Column == TextProfile.UnseenColumn("t0") && f.Level >= DriftLevel.Moderate) ? 1 : 0;
            }
    }
    string R(int n) => (n / (double)total).ToString("0.000", CultureInfo.InvariantCulture);
    for (var s = 0; s < shifts.Length; s++)
        Console.WriteLine($"| {shifts[s].Name} | {R(counts[s, 0])} | {R(counts[s, 1])} | {R(counts[s, 2])} | {R(counts[s, 3])} |");
    return 0;
}

List<IReadOnlyDictionary<string, object?>> Window(Func<Random, int, IReadOnlySet<int>, double, string[]> row, Random rng, IReadOnlySet<int> moved, double scale)
    => [.. Enumerable.Range(0, windowRows).Select(_ =>
    {
        var cells = row(rng, columns, moved, scale);
        return (IReadOnlyDictionary<string, object?>)Enumerable.Range(0, columns).ToDictionary(j => $"c{j}", j => (object?)cells[j]);
    })];

static string? TextOf(object? cell) => cell as string;

static double Moved(IReadOnlySet<int> moved, int column, double amount) => moved.Contains(column) ? amount : 0;

static string[] RowText(int columns, Func<int, string> draw)
    => [.. Enumerable.Range(0, columns).Select(draw)];

static string[] RowOrEmpty(Random rng, int columns, double missing, Func<int, double> draw)
    => [.. Enumerable.Range(0, columns).Select(j => rng.NextDouble() < missing ? "" : draw(j).ToString("R", CultureInfo.InvariantCulture))];

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
