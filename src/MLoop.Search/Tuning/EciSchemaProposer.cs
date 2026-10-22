namespace MLoop.Search.Tuning;

/// <summary>
/// Chooses which schema (trainer × featurization) the next trial searches, by estimated cost for
/// improvement (ECI, FLAML): the cheapest schema first, then each with probability proportional to
/// 1 / ECI, where a schema's ECI grows with the time it has spent without improving and with how far it
/// trails the best.
/// </summary>
/// <remarks>
/// Ported from ML.NET AutoML (<c>Tuner/PipelineProposer.cs</c>, dotnet/machinelearning <c>cf9093abb6</c>),
/// draws in the same order. One upstream rule is not kept: a trial counts as failed there when it took
/// under a second (<c>DurationInMilliseconds / 1000</c> is integer division, so 0), which sent every fast
/// trainer's ECI to twice the maximum on small data. Here a trial fails only when its loss is not finite.
/// <para>
/// Upstream feeds it each trial's measured seconds. A cost here is whatever the caller measures in, as
/// long as one search uses one unit — only ratios enter the estimate. <see cref="TrialRunner"/> passes the
/// schema's relative cost rather than the clock, so which schema a seed picks does not depend on how
/// busy the machine was.
/// </para>
/// </remarks>
public sealed class EciSchemaProposer
{
    private readonly string[] _schemas;
    private readonly Dictionary<string, double> _initialCost;
    private readonly Dictionary<string, double> _eci;
    private readonly Random _rand;
    private Dictionary<string, double>? _k1, _k2, _e1, _e2;
    private double _globalBestError;
    private bool _initialized;

    /// <param name="schemas">The schemas and their initial cost estimates, in search order.</param>
    /// <param name="seed">The seed; <see langword="null"/> varies run to run.</param>
    public EciSchemaProposer(IReadOnlyList<(string Schema, double InitialCost)> schemas, int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(schemas);
        if (schemas.Count == 0)
            throw new ArgumentException("At least one schema is needed.", nameof(schemas));
        _schemas = [.. schemas.Select(s => s.Schema)];
        _initialCost = schemas.ToDictionary(s => s.Schema, s => s.InitialCost, StringComparer.Ordinal);
        _eci = schemas.ToDictionary(s => s.Schema, s => s.InitialCost, StringComparer.Ordinal);
        _rand = seed is { } s ? new Random(s) : new Random();
    }

    /// <summary>Each schema's current ECI.</summary>
    public IReadOnlyDictionary<string, double> Eci => _eci;

    /// <summary>The schema for the next trial.</summary>
    public string Propose() => Propose(_ => true)!;

    /// <summary>
    /// The schema for the next trial among those <paramref name="allowed"/> admits, with the others' share
    /// spread over them — or <see langword="null"/> when it admits none. One draw either way, so the same
    /// admitted sets give the same schemas.
    /// </summary>
    public string? Propose(Func<string, bool> allowed)
    {
        ArgumentNullException.ThrowIfNull(allowed);
        var candidates = _schemas.Where(allowed).ToArray();
        if (candidates.Length == 0)
            return null;

        if (!_initialized)
        {
            _initialized = true;
            return candidates.OrderBy(id => _eci[id]).First();
        }

        var probabilities = candidates.Select(id => (1 / _eci[id]) + double.Epsilon).ToArray();
        var total = probabilities.Sum();
        var draw = _rand.NextDouble();
        var sum = 0.0;
        int i;
        for (i = 0; i != candidates.Length; ++i)
        {
            sum += probabilities[i] / total;
            if (sum > draw)
                break;
        }
        // Rounding can leave the cumulative sum a hair under the draw; the last schema then takes it.
        return candidates[Math.Min(i, candidates.Length - 1)];
    }

    /// <summary>Takes a trial's result for <paramref name="schema"/>: its <paramref name="loss"/> (to minimize) and <paramref name="cost"/>.</summary>
    public void Update(string schema, double loss, double cost)
    {
        _initialized = true;
        if (!double.IsFinite(loss) || !(cost >= 0))
        {
            _eci[schema] = _eci.Values.Max() * 2;
            return;
        }

        if (_k1 is null)
        {
            _k1 = _schemas.ToDictionary(id => id, id => cost * _initialCost[id] / _initialCost[schema], StringComparer.Ordinal);
            _k2 = new Dictionary<string, double>(_k1, StringComparer.Ordinal);
            _e1 = _schemas.ToDictionary(id => id, _ => loss, StringComparer.Ordinal);
            _e2 = _schemas.ToDictionary(id => id, _ => loss + 0.05 * Math.Abs(loss), StringComparer.Ordinal);
            _globalBestError = loss;
        }
        else if (loss >= _e1![schema])
        {
            _k1[schema] += cost;
        }
        else
        {
            _k2![schema] = _k1[schema];
            _k1[schema] = cost;
            _e2![schema] = _e1[schema];
            _e1[schema] = loss;
            _globalBestError = Math.Min(_globalBestError, loss);
        }

        var eci1 = Math.Max(_k1[schema], _k2![schema]);
        var breakThrough = (2 * (loss - _globalBestError) + double.Epsilon)
                           / ((_e2![schema] - _e1![schema]) / (_k2[schema] + _k1[schema]) + double.Epsilon);
        _eci[schema] = Math.Max(eci1, breakThrough);
    }
}
