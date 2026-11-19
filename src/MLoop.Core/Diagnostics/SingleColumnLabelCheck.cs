using System.Globalization;
using MLoop.Core.Data;

namespace MLoop.Core.Diagnostics;

/// <summary>One column that alone reproduces the label: its score, and what the label's own most common value (or mean) gets.</summary>
/// <param name="Column">Column name as the file writes it.</param>
/// <param name="Score">Classification: how often the column alone names the label (leave-one-out). Regression: share of the label's variance it explains.</param>
/// <param name="Baseline">Classification: the share of rows the most common class holds. Regression: 0.</param>
/// <param name="Reduction">Classification: the share of the most-common-class errors the column removes (λ). Regression: same as <paramref name="Score"/>.</param>
/// <param name="IsRegression">Which of the two readings the numbers carry.</param>
public sealed record SingleColumnFinding(string Column, double Score, double Baseline, double Reduction, bool IsRegression);

/// <summary>
/// A column that by itself nearly is the label. A column computed from the label, or recorded after it, scores almost as well
/// as the label itself in the data the model is trained on and is missing — or meaningless — where the model is used; the held-out
/// score cannot tell, and stays under the ≥ 0.999 warning once the copy carries a little noise. This looks at each column alone,
/// on the whole file before the split (like the exclusions), and reports the ones past a threshold. It only reports: nothing is
/// excluded and no promotion rule reads it.
/// <para>
/// Classification reads each column's levels and predicts every row from the <i>other</i> rows of its level (leave-one-out, so a column
/// of values that differ on every row — an identifier — has nothing to predict from and cannot look pure); the score is the share of
/// the most-common-class errors this removes (Goodman–Kruskal λ). Regression reads the larger of the column's correlation ratio over
/// its levels (a numeric column cut into ten bins) and, for a numeric column, its squared rank correlation — bins lose the variance
/// inside a bin, so a continuous copy never reads 1 by bins alone. The thresholds come from a measurement over 216 public tables (classification:
/// 3 of 111 read above 0.75, all with one genuinely strong column; regression: 1 of 67 above 0.95, a physical law between two measurements)
/// and over copies planted into 25 tables.
/// </para>
/// </summary>
public static class SingleColumnLabelCheck
{
    /// <summary>λ at or above which a classification column is reported — 3 of 111 public tables read above it, all with one genuinely strong column.</summary>
    public const double ClassificationThreshold = 0.75;

    /// <summary>R² at or above which a regression column is reported — 1 of 67 public tables read above it (a physical law between two measurements).</summary>
    public const double RegressionThreshold = 0.95;

    /// <summary>Fewer rows than this and a column's levels have too few rows to predict one another.</summary>
    public const int MinRows = 100;

    /// <summary>Rows read: more are sampled (uniformly, fixed seed) — the statistic needs a table's shape, not every row.</summary>
    public const int MaxRows = 100_000;

    /// <summary>
    /// Cells (rows × columns) held at once. A table of two thousand columns at a hundred thousand rows would otherwise be two hundred
    /// million strings; the sample shrinks with the width instead (a 2,049-column table takes 3,900 rows — enough for a column's levels).
    /// </summary>
    public const int MaxCells = 8_000_000;

    /// <summary>How many rows to keep for a table of this many columns.</summary>
    internal static int RowCap(int columns) => Math.Clamp(MaxCells / Math.Max(1, columns), MinRows * 4, MaxRows);

    private const int Bins = 10;
    private const int MaxLevelsAsIs = 20;

    /// <summary>Tasks this reads the label of: the ones whose model reads feature columns and has a label to copy.</summary>
    public static bool Applies(string? task)
    {
        var canonical = Models.TaskTypes.Canonical(task ?? "");
        return canonical is "binary-classification" or "multiclass-classification" or "regression";
    }

    /// <summary>
    /// The columns of <paramref name="path"/> past the threshold, strongest first. <paramref name="skip"/> names columns that are not
    /// features (excluded, ignored, weights); the label is skipped on its own. Returns empty when the file cannot be read or is too small.
    /// </summary>
    public static IReadOnlyList<SingleColumnFinding> Find(string path, string label, string? task, IEnumerable<string>? skip = null)
    {
        if (!Applies(task) || string.IsNullOrEmpty(label) || !File.Exists(path)) return [];
        try
        {
            return FindCore(path, label, Models.TaskTypes.Canonical(task!) == "regression", skip);
        }
        catch
        {
            return []; // an advisory; reading failures belong to the load that follows
        }
    }

    private static IReadOnlyList<SingleColumnFinding> FindCore(string path, string label, bool regression, IEnumerable<string>? skip)
    {
        string[]? header = null;
        var rows = new List<string[]>();
        var rng = new Random(20261119);
        int seen = 0, cap = MaxRows;
        foreach (var record in TableRows.ReadRecords(path, int.MaxValue, includeHeader: true))
        {
            if (header is null) { header = record; cap = RowCap(header.Length); continue; }
            if (record.Length < header.Length) continue;
            seen++;
            if (rows.Count < cap) rows.Add(record);
            else
            {
                int j = rng.Next(seen);
                if (j < cap) rows[j] = record;
            }
        }
        if (header is null) return [];

        int labelIndex = Array.FindIndex(header, h => h.Trim().Equals(label, StringComparison.OrdinalIgnoreCase));
        if (labelIndex < 0) return [];

        // Rows whose label is missing are not used (they are dropped before training too).
        var usable = rows.Where(r => !string.IsNullOrWhiteSpace(r[labelIndex])).ToList();
        double[]? target = null;
        int[]? classes = null;
        if (regression)
        {
            var keep = new List<string[]>(usable.Count);
            var values = new List<double>(usable.Count);
            foreach (var r in usable)
            {
                if (double.TryParse(r[labelIndex].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var v) && double.IsFinite(v))
                {
                    keep.Add(r);
                    values.Add(v);
                }
            }
            usable = keep;
            target = [.. values];
        }
        if (usable.Count < MinRows) return [];
        if (!regression) classes = Codes(usable.Select(r => r[labelIndex].Trim()));
        if (regression && Variance(target!) <= 0) return [];
        if (!regression && classes!.Max() < 1) return [];

        var skipSet = new HashSet<string>(skip ?? [], StringComparer.OrdinalIgnoreCase) { label };
        var findings = new List<SingleColumnFinding>();
        for (int c = 0; c < header.Length; c++)
        {
            var name = header[c].Trim();
            if (string.IsNullOrEmpty(name) || skipSet.Contains(name) || CsvDataLoader.IsLikelyIndexColumn(name)) continue;

            var raw = usable.Select(r => r[c].Trim()).ToArray();
            var (levels, numeric) = LevelsOf(raw);
            if (levels.Max() < 1) continue;

            if (regression)
            {
                var r2 = AdjustedCorrelationRatio(levels, target!);
                if (numeric is not null)
                    r2 = Math.Max(r2, SquaredRankCorrelation(numeric, target!));
                if (r2 >= RegressionThreshold)
                    findings.Add(new SingleColumnFinding(name, r2, 0, r2, true));
            }
            else
            {
                var (accuracy, baseline, lambda) = LeaveOneOutReduction(levels, classes!);
                if (lambda >= ClassificationThreshold)
                    findings.Add(new SingleColumnFinding(name, accuracy, baseline, lambda, false));
            }
        }
        return [.. findings.OrderByDescending(f => f.Reduction)];
    }

    /// <summary>
    /// Level codes of a column: a numeric column (≥ 80 % of its non-empty values read as numbers) with more than twenty distinct values
    /// is cut at its deciles — <b>by value</b>, so equal values share a bin (cutting by rank splits a run of ties by row order, and a
    /// file sorted by label then hands the label to every mostly-constant column); any other column is its distinct strings. An empty
    /// value is a level of its own. The parsed numbers come back for a numeric column.
    /// </summary>
    internal static (int[] Levels, double[]? Numeric) LevelsOf(string[] raw)
    {
        int n = raw.Length;
        var parsed = new double[n];
        int nonEmpty = 0, numbers = 0;
        for (int i = 0; i < n; i++)
        {
            if (raw[i].Length == 0) { parsed[i] = double.NaN; continue; }
            nonEmpty++;
            if (double.TryParse(raw[i], NumberStyles.Float, CultureInfo.InvariantCulture, out var v) && double.IsFinite(v))
            {
                parsed[i] = v;
                numbers++;
            }
            else parsed[i] = double.NaN;
        }

        bool numeric = nonEmpty > 0 && numbers >= 0.8 * nonEmpty;
        if (numeric)
        {
            var sorted = parsed.Where(v => !double.IsNaN(v)).OrderBy(v => v).ToArray();
            if (sorted.Distinct().Count() > MaxLevelsAsIs)
            {
                var cuts = Enumerable.Range(1, Bins - 1)
                    .Select(k => sorted[(int)((long)k * sorted.Length / Bins)])
                    .Distinct().ToArray();
                var levels = new int[n];
                for (int i = 0; i < n; i++)
                {
                    if (double.IsNaN(parsed[i])) { levels[i] = cuts.Length + 1; continue; }
                    int lo = 0, hi = cuts.Length;       // number of cuts the value is at or above
                    while (lo < hi)
                    {
                        int mid = (lo + hi) / 2;
                        if (parsed[i] >= cuts[mid]) lo = mid + 1; else hi = mid;
                    }
                    levels[i] = lo;
                }
                return (Compact(levels), parsed);
            }
        }
        return (Codes(raw), numeric ? parsed : null);
    }

    private static int[] Codes(IEnumerable<string> values)
    {
        var map = new Dictionary<string, int>(StringComparer.Ordinal);
        return [.. values.Select(v =>
        {
            if (!map.TryGetValue(v, out var code)) map[v] = code = map.Count;
            return code;
        })];
    }

    private static int[] Compact(int[] levels)
    {
        var map = new Dictionary<int, int>();
        return [.. levels.Select(l =>
        {
            if (!map.TryGetValue(l, out var code)) map[l] = code = map.Count;
            return code;
        })];
    }

    /// <summary>Accuracy of predicting each row's class from the other rows of its level, the most-common-class share, and λ between them.</summary>
    internal static (double Accuracy, double Baseline, double Lambda) LeaveOneOutReduction(int[] levels, int[] classes)
    {
        int n = classes.Length;
        int levelCount = levels.Max() + 1, classCount = classes.Max() + 1;
        var counts = new int[levelCount, classCount];
        var classTotals = new int[classCount];
        for (int i = 0; i < n; i++)
        {
            counts[levels[i], classes[i]]++;
            classTotals[classes[i]]++;
        }

        double baseline = (double)classTotals.Max() / n;
        if (baseline >= 1) return (baseline, baseline, 0);

        long correct = 0;
        for (int l = 0; l < levelCount; l++)
        {
            int total = 0;
            for (int k = 0; k < classCount; k++) total += counts[l, k];
            if (total < 2) continue;            // nothing to predict it from: counted wrong
            for (int k = 0; k < classCount; k++)
            {
                int ck = counts[l, k];
                if (ck == 0) continue;
                int bestOther = 0;
                for (int j = 0; j < classCount; j++)
                    if (j != k && counts[l, j] > bestOther) bestOther = counts[l, j];
                if (ck - 1 >= bestOther) correct += ck;     // the row's own class is (at least tied for) the other rows' majority
            }
        }

        double accuracy = (double)correct / n;
        return (accuracy, baseline, Math.Max(0, (accuracy - baseline) / (1 - baseline)));
    }

    /// <summary>Share of the target's variance the column's levels explain, corrected for the degrees of freedom the levels use.</summary>
    internal static double AdjustedCorrelationRatio(int[] levels, double[] target)
    {
        int n = target.Length;
        int k = levels.Max() + 1;
        var sums = new double[k];
        var counts = new int[k];
        double total = 0;
        for (int i = 0; i < n; i++)
        {
            sums[levels[i]] += target[i];
            counts[levels[i]]++;
            total += target[i];
        }

        double mean = total / n;
        double between = 0;
        for (int l = 0; l < k; l++)
            if (counts[l] > 0) between += counts[l] * Math.Pow(sums[l] / counts[l] - mean, 2);
        double all = target.Sum(v => (v - mean) * (v - mean));
        if (all <= 0) return 0;

        int used = counts.Count(c => c > 0);
        if (n <= used) return 0;
        double ratio = between / all;
        return Math.Max(0, 1 - (1 - ratio) * (n - 1) / (n - used));
    }

    /// <summary>Squared Spearman correlation between a numeric column and the target, over the rows where the column has a number.</summary>
    internal static double SquaredRankCorrelation(double[] column, double[] target)
    {
        var index = Enumerable.Range(0, column.Length).Where(i => !double.IsNaN(column[i])).ToArray();
        if (index.Length < 3) return 0;
        var a = Ranks(index.Select(i => column[i]).ToArray());
        var b = Ranks(index.Select(i => target[i]).ToArray());
        double ma = a.Average(), mb = b.Average(), sab = 0, saa = 0, sbb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sab += (a[i] - ma) * (b[i] - mb);
            saa += (a[i] - ma) * (a[i] - ma);
            sbb += (b[i] - mb) * (b[i] - mb);
        }
        if (saa <= 0 || sbb <= 0) return 0;
        double rho = sab / Math.Sqrt(saa * sbb);
        return rho * rho;
    }

    /// <summary>Average ranks (equal values share the mean of the ranks they span).</summary>
    private static double[] Ranks(double[] values)
    {
        var order = Enumerable.Range(0, values.Length).OrderBy(i => values[i]).ToArray();
        var ranks = new double[values.Length];
        for (int i = 0; i < order.Length;)
        {
            int j = i;
            while (j + 1 < order.Length && values[order[j + 1]] == values[order[i]]) j++;
            double rank = (i + j) / 2.0 + 1;
            for (int t = i; t <= j; t++) ranks[order[t]] = rank;
            i = j + 1;
        }
        return ranks;
    }

    private static double Variance(double[] values)
    {
        double mean = values.Average();
        return values.Sum(v => (v - mean) * (v - mean)) / values.Length;
    }
}
