namespace MLoop.Search.Tuning;

/// <summary>
/// FLOW2 — the local search inside one schema of the cost-frugal tuner (CFO, FLAML): from an
/// incumbent point it tries a random direction <c>x + u</c>, then <c>x − u</c>; an improvement moves
/// the incumbent there and widens the step, a run of non-improving tries narrows it, and a step below
/// 1e-4 is convergence.
/// </summary>
/// <remarks>
/// Ported from ML.NET AutoML (<c>Tuner/Flow2.cs</c>, dotnet/machinelearning <c>cf9093abb6</c>) step for
/// step, including the order random numbers are drawn in. Two of its choices are kept because the
/// search MLoop ran until now made them, and changing one is measured on its own:
/// <list type="bullet">
/// <item>the step's scale <c>d</c> is the number of <b>top-level</b> entries of the search space —
/// AutoML nests every pipeline under one key, so in practice <c>d = 1</c> (initial step 0.1, at most 1,
/// non-improving window 2), not the number of hyperparameters;</item>
/// <item>the non-improving window is <c>2d</c> for the first incumbent and 2 after a reset.</item>
/// </list>
/// One is not kept: upstream clamps a coordinate above 1 to <c>0.99999999</c>, so an integer's maximum
/// and a choice's last option can never be proposed from a step that crosses the edge; here it clamps
/// to 1 (<see cref="UpstreamUpperClamp"/> holds the upstream value for the trace tests).
/// </remarks>
public sealed class Flow2
{
    private const double StepSize = 0.1;
    private const double StepLowerBound = 0.0001;

    /// <summary>Upstream's clamp for a coordinate above 1 — kept for trace comparisons only.</summary>
    internal const double UpstreamUpperClamp = 0.99999999;

    private readonly SearchSpace _space;
    private readonly TunerRandom _rng;
    private readonly double _convergeSpeed;
    private readonly int _stepDimensions;
    private readonly double _stepUpperBound;
    private readonly double _upperClamp;
    private double _step;
    private double[]? _directionTried;
    private double[] _incumbent;
    private int _numAllowed4Incumbent;

    /// <param name="space">The space searched.</param>
    /// <param name="initial">The first incumbent.</param>
    /// <param name="rng">The random numbers — shared with the tuner and its other threads.</param>
    /// <param name="stepDimensions">The step's scale <c>d</c> (see remarks).</param>
    /// <param name="convergeSpeed">How much a step widens on improvement and narrows on a stall.</param>
    /// <param name="upperClamp">Where a coordinate above 1 is clamped.</param>
    public Flow2(SearchSpace space, SearchPoint initial, TunerRandom rng, int stepDimensions = 1, double convergeSpeed = 1.5, double upperClamp = 1.0)
    {
        _space = space ?? throw new ArgumentNullException(nameof(space));
        _rng = rng ?? throw new ArgumentNullException(nameof(rng));
        if (stepDimensions < 1)
            throw new ArgumentOutOfRangeException(nameof(stepDimensions));
        BestConfig = initial ?? throw new ArgumentNullException(nameof(initial));
        _incumbent = space.UnitOf(initial);
        _stepDimensions = stepDimensions;
        _numAllowed4Incumbent = 2 * stepDimensions;
        _step = Math.Min(StepSize * Math.Sqrt(stepDimensions), Math.Sqrt(stepDimensions));
        _stepUpperBound = Math.Sqrt(stepDimensions);
        _convergeSpeed = convergeSpeed;
        _upperClamp = upperClamp;
    }

    /// <summary>The best loss seen, or <see langword="null"/> before the first result.</summary>
    public double? BestLoss { get; private set; }

    /// <summary>The cost (ms) of the incumbent's trial.</summary>
    public double? CostIncumbent { get; private set; }

    /// <summary>The incumbent.</summary>
    public SearchPoint BestConfig { get; private set; }

    /// <summary>Whether the step has shrunk below its lower bound.</summary>
    public bool IsConverged => _step < StepLowerBound;

    /// <summary>A new search from <paramref name="config"/>, sharing this one's space and random numbers.</summary>
    public SearchThread CreateSearchThread(SearchPoint config, double loss, double cost)
    {
        var flow2 = new Flow2(_space, config, _rng, _stepDimensions, _convergeSpeed, _upperClamp)
        {
            BestLoss = loss,
            CostIncumbent = cost,
        };
        return new SearchThread(flow2);
    }

    /// <summary>The next point to try.</summary>
    public SearchPoint Suggest()
    {
        _numAllowed4Incumbent -= 1;
        double[] move;
        if (_directionTried is { } tried)
        {
            move = [.. _incumbent.Zip(tried, (x, u) => x + (u * -1))];
            _directionTried = null;
        }
        else
        {
            _directionTried = RandomDirection();
            move = [.. _incumbent.Zip(_directionTried, (x, u) => x + u)];
        }
        return _space.PointAt(Project(move));
    }

    /// <summary>Takes a trial's result: an improvement becomes the incumbent; a stall narrows the step.</summary>
    public void ReceiveTrialResult(SearchPoint point, double loss, double cost)
    {
        if (BestLoss is null || loss < BestLoss)
        {
            BestLoss = loss;
            BestConfig = point;
            _incumbent = _space.UnitOf(point);
            CostIncumbent = cost;
            _numAllowed4Incumbent = 2 * _stepDimensions;
            _step = Math.Min(_step * _convergeSpeed, _stepUpperBound);
            _directionTried = null;
            return;
        }
        if (_numAllowed4Incumbent == 0)
        {
            _numAllowed4Incumbent = 2;
            if (!IsConverged)
                _step /= _convergeSpeed;
        }
    }

    private double[] RandomDirection()
    {
        var vec = _rng.Normal(0, 1, _space.FeatureDim);
        double s = 0;
        foreach (var x in vec)
            s += x * x;
        var scale = _step / Math.Sqrt(s);
        return [.. vec.Select(x => x * scale)];
    }

    private double[] Project(double[] move) => [.. move.Select(x => x < 0 ? 0 : x > 1 ? _upperClamp : x)];
}

/// <summary>
/// One FLOW2 search and its progress bookkeeping — the unit the cost-frugal tuner runs and replaces
/// when it converges.
/// </summary>
public sealed class SearchThread
{
    internal SearchThread(Flow2 search) => Search = search;

    /// <summary>The local search.</summary>
    public Flow2 Search { get; }

    /// <summary>Whether its search has converged.</summary>
    public bool IsConverged => Search.IsConverged;

    /// <summary>Takes a trial's result.</summary>
    public void OnTrialComplete(SearchPoint point, double loss, double cost) => Search.ReceiveTrialResult(point, loss, cost);

    /// <summary>The next point to try.</summary>
    public SearchPoint Suggest() => Search.Suggest();
}
