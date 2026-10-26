namespace MLoop.Search.Tuning;

/// <summary>
/// The cost-frugal tuner (CFO, FLAML) for one schema's search space: it starts at the space's default —
/// cheap on purpose (4 trees, 4 leaves) — and grows from there with a <see cref="Flow2"/> local search,
/// restarting when the search converges. Given starting points (a trainer's own defaults), it tries them after
/// the cheap default and begins the local search from the best of the two.
/// </summary>
/// <remarks>
/// Ported from ML.NET AutoML (<c>Tuner/CostFrugalTuner.cs</c> · <c>SearchThread.cs</c>, dotnet/machinelearning
/// <c>cf9093abb6</c>), random draws in the same order. A restart draws its start uniformly from the box of
/// configurations seen — which upstream widens only with the first trial's point, so in practice every
/// restart begins at the first configuration again. Kept as upstream does it until a change is measured.
/// </remarks>
public sealed class CostFrugalTuner
{
    private readonly SearchSpace _space;
    private readonly TunerRandom _rng;
    private readonly Flow2 _localSearch;
    private readonly Dictionary<int, SearchThread> _threads = [];
    private readonly Dictionary<int, int> _proposedBy = [];
    private readonly double[] _boundMin;
    private readonly double[] _boundMax;
    private readonly Queue<SearchPoint> _starts;
    private readonly HashSet<int> _startTrials = [];
    private (SearchPoint Point, double Loss, double Cost)? _bestStart;
    private (SearchPoint Point, double Loss, double Cost)? _defaultResult;
    private bool _defaultProposed;
    private int _defaultTrial = -1;
    private int _currentThread;
    private bool _initUsed;
    private double _bestLoss = double.MaxValue;

    /// <param name="space">The space searched.</param>
    /// <param name="seed">The seed; <see langword="null"/> varies run to run.</param>
    /// <param name="stepDimensions">FLOW2's step scale (see <see cref="Flow2"/>; 1 as AutoML ran it).</param>
    /// <param name="startingPoints">Points tried after the cheap default, in order (none: as AutoML ran it).</param>
    public CostFrugalTuner(SearchSpace space, int? seed = null, int stepDimensions = 1, IReadOnlyList<SearchPoint>? startingPoints = null)
        : this(space, new TunerRandom(seed), stepDimensions, upperClamp: 1.0, startingPoints)
    {
    }

    internal CostFrugalTuner(SearchSpace space, TunerRandom rng, int stepDimensions, double upperClamp, IReadOnlyList<SearchPoint>? startingPoints = null)
    {
        _space = space ?? throw new ArgumentNullException(nameof(space));
        _rng = rng;
        _starts = new Queue<SearchPoint>(startingPoints ?? []);
        var initial = space.PointAt(space.Default);
        _boundMin = space.UnitOf(initial);
        _boundMax = space.UnitOf(initial);
        _localSearch = new Flow2(space, initial, rng, stepDimensions, upperClamp: upperClamp);
    }

    /// <summary>The best point seen so far.</summary>
    public SearchPoint? BestConfig { get; private set; }

    /// <summary>The point for trial <paramref name="trialId"/>.</summary>
    public SearchPoint Propose(int trialId)
    {
        _proposedBy[trialId] = _currentThread;
        if (_threads.Count == 0 && !_initUsed)
        {
            // The cheap default goes first, so a search's first trial is its cheapest — a starting point can be
            // a hundred trees on a table where that takes minutes, and the first trial is the one that always finishes.
            if (!_defaultProposed)
            {
                _defaultProposed = true;
                _defaultTrial = trialId;
            }
            else if (_starts.Count > 0)
            {
                _startTrials.Add(trialId);
                return _starts.Dequeue();
            }
        }

        return _initUsed
            ? _threads[_currentThread].Suggest()
            : _space.PointAt([.. _boundMin.Select((min, i) => _rng.Uniform(min, _boundMax[i]))]);
    }

    /// <summary>
    /// Takes trial <paramref name="trialId"/>'s result — <paramref name="loss"/> to minimize (a failed
    /// trial: <see cref="double.MaxValue"/>) and its <paramref name="cost"/> (any unit, one per search — it is
    /// recorded for the incumbent and steers nothing).
    /// </summary>
    public void Update(int trialId, SearchPoint point, double loss, double cost)
    {
        ArgumentNullException.ThrowIfNull(point);
        if (loss < _bestLoss)
        {
            BestConfig = point;
            _bestLoss = loss;
        }

        if (_startTrials.Remove(trialId))
        {
            if (loss < (_bestStart?.Loss ?? double.MaxValue))
                _bestStart = (point, loss, cost);
        }
        else if (trialId == _defaultTrial)
        {
            _defaultResult = (point, loss, cost);
        }
        else if (_threads.Count == 0)
        {
            // A restart's first trial: the local search begins where it was drawn.
            _threads[_currentThread] = _localSearch.CreateSearchThread(point, loss, cost);
            _initUsed = true;
            Widen(_space.UnitOf(point));
            return;
        }
        else
        {
            var thread = _proposedBy.TryGetValue(trialId, out var proposer) ? proposer : _currentThread;
            _threads[thread].OnTrialComplete(point, loss, cost);
            if (_threads[thread].IsConverged)
            {
                _threads.Remove(thread);
                _currentThread += 1;
                _initUsed = false;
            }

            return;
        }

        // The cheap default and every starting point have answered: the local search begins at the best of them
        // — a starting point that beat the default is where the walk is worth taking.
        if (_defaultResult is { } cheap && _starts.Count == 0 && _startTrials.Count == 0)
        {
            var (from, fromLoss, fromCost) = _bestStart is { } start && start.Loss < cheap.Loss ? start : cheap;
            _bestStart = null;
            _threads[_currentThread] = _localSearch.CreateSearchThread(from, fromLoss, fromCost);
            _initUsed = true;
            Widen(_space.UnitOf(from));
        }
    }

    private void Widen(double[] unit)
    {
        for (var i = 0; i < unit.Length; i++)
        {
            if (unit[i] < _boundMin[i])
                _boundMin[i] = unit[i];
            else if (unit[i] > _boundMax[i])
                _boundMax[i] = unit[i];
        }
    }
}
