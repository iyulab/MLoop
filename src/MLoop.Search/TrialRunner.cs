using System.Diagnostics;
using System.Globalization;
using Microsoft.ML;
using MLoop.Search.Candidates;
using MLoop.Search.Tuning;
using MLoop.Search.Validation;

namespace MLoop.Search;

/// <summary>What a search needs besides the data: the trainers, the metric, the scorer, the budget.</summary>
/// <param name="Candidates">The trainers to try (<see cref="CandidateCatalog"/>).</param>
/// <param name="Metric">The metric trials are ranked by.</param>
/// <param name="Score">The metrics of a scored validation part — the host's metric authority, weights included.</param>
/// <param name="Budget">Wall-clock time for the search, the refit of the best included.</param>
/// <param name="Settings">Seed and threads.</param>
/// <param name="Weight">The column that weights rows, if any — trained on, never a feature.</param>
/// <param name="Stratify">Whether folds keep the label's proportions (classification).</param>
/// <param name="MaxTrials">At most this many trials — for comparing two searches trial for trial; <see langword="null"/> for the budget alone.</param>
/// <param name="Preprocess">
/// Steps fitted ahead of the featurizer in every trial — on each fold's training rows only, so a statistic
/// such as a column's mean never sees the rows that score it — and kept in the refit model. They transform
/// columns in place: a column keeps its name and its type.
/// </param>
public sealed record SearchRequest(
    IReadOnlyList<TrainerCandidate> Candidates,
    IMetricPolicy Metric,
    Func<IDataView, IReadOnlyDictionary<string, double>> Score,
    TimeSpan Budget,
    SearchSettings Settings,
    string? Weight = null,
    bool Stratify = true,
    int? MaxTrials = null,
    IEstimator<ITransformer>? Preprocess = null);

/// <summary>A finished search: every trial, the best, and the best refit on the whole partition.</summary>
/// <param name="Trials">Every trial in the order it finished, failures included.</param>
/// <param name="Best">The best trial by the metric.</param>
/// <param name="BestValidation">The best trial's fold scores.</param>
/// <param name="Model">The best pipeline refit on the whole training partition — or, when that refit failed, its first fold's model.</param>
/// <param name="FallbackReason">Why <see cref="Model"/> is a fold's model, or <see langword="null"/>.</param>
/// <param name="Folds">How the partition was validated.</param>
public sealed record SearchResult(
    IReadOnlyList<TrialResult> Trials,
    TrialResult Best,
    ValidationScore BestValidation,
    ITransformer Model,
    string? FallbackReason,
    FoldPlan Folds);

/// <summary>No trial of a search produced a score. <see cref="Trials"/> says why each failed.</summary>
public sealed class SearchFailedException(IReadOnlyList<TrialResult> trials, string message) : Exception(message)
{
    /// <summary>The trials run, every one failed.</summary>
    public IReadOnlyList<TrialResult> Trials { get; } = trials;
}

/// <summary>
/// Runs a search: ECI picks the schema (trainer × categorical encoding), its cost-frugal tuner picks the
/// settings, the validator scores the pipeline as a mean over folds, and the best is refit on the whole
/// training partition.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item><b>The first trial always finishes</b>, whatever the budget — a search returns a model.</item>
/// <item><b>The refit is budgeted</b>: the search stops starting trials once the time left is what refitting
/// the best is expected to take (its mean fold fit × folds ⁄ (folds − 1)), and a trial still running at that
/// point stops inside its fit and is dropped — it is neither a result nor a failure.</item>
/// <item><b>A trial that cannot finish is not started.</b> Past the first success, a schema whose last trial
/// took longer than the time left before the refit is not proposed — the tuner grows a schema's settings
/// from there, so its next trial would take at least as long, be cut at the deadline, and be dropped. The draw goes to the schemas that still fit, so the tail of the budget is spent on cheaper
/// trainers; the search ends when none fits.</item>
/// <item><b>A failed trial is recorded and the search goes on</b>; the tuner reads it as the worst loss.</item>
/// <item><b>Cancellation by the caller stops at once</b> and throws; budget expiry keeps the best.</item>
/// <item><b>A seed fixes the search; the clock only ends it.</b> The tuners and the schema proposer read each
/// trial's loss and its schema's relative cost (<see cref="TrainerCandidate.InitialCost"/>), never its measured
/// time — so the same seed on the same data proposes the same trials in the same order on any machine, and a
/// budget decides how many of them run. Measured time goes to <see cref="TrialResult.RuntimeSeconds"/>, the
/// refit reserve, and — near the end of a budget — which schemas still fit; there, and only there, a busy
/// machine can draw differently from an idle one.</item>
/// </list>
/// </remarks>
public sealed class TrialRunner
{
    private readonly SearchRequest _request;
    private readonly ITrialSink? _sink;

    public TrialRunner(SearchRequest request, ITrialSink? sink = null)
    {
        _request = request ?? throw new ArgumentNullException(nameof(request));
        if (request.Candidates.Count == 0)
            throw new ArgumentException("A search needs at least one candidate trainer.", nameof(request));
        _sink = sink;
    }

    /// <summary>Searches <paramref name="train"/> (the training partition) for <paramref name="roles"/>' label.</summary>
    public SearchResult Run(IDataView train, ColumnRoles roles, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(train);
        ArgumentNullException.ThrowIfNull(roles);
        var clock = Stopwatch.StartNew();
        var settings = _request.Settings;
        var plan = FoldPlan.For(train, roles.Label, _request.Stratify, settings.Seed);
        var columns = new TrainerColumns(roles.Label, "Features", _request.Weight);
        var withMissingValues = Featurizer.ColumnsWithMissingValues(train, roles);

        var encodings = roles.Of(ColumnRole.Categorical).Count > 0
            ? new[] { CategoricalEncoding.OneHot, CategoricalEncoding.Hash }
            : [CategoricalEncoding.OneHot];
        var schemas = _request.Candidates
            .SelectMany(c => encodings.Select(e => (Id: encodings.Length == 1 || e == CategoricalEncoding.OneHot ? c.Name : $"{c.Name}+hash", Candidate: c, Encoding: e)))
            .ToList();
        var proposer = new EciSchemaProposer([.. schemas.Select(s => (s.Id, s.Candidate.InitialCost))], settings.Seed);
        var tuners = schemas.ToDictionary(s => s.Id, s => new CostFrugalTuner(s.Candidate.Space, settings.Seed), StringComparer.Ordinal);

        var trials = new List<TrialResult>();
        var lastSeconds = new Dictionary<string, double>(StringComparer.Ordinal);
        (TrialResult Trial, double Value, ValidationScore Score, IEstimator<ITransformer> Pipeline, MLContext Context, double FitSeconds)? best = null;

        for (var trialId = 0; ; trialId++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var reserve = best is { } b ? b.FitSeconds * plan.Folds / Math.Max(plan.Folds - 1, 1) : 0;
            var deadline = _request.Budget.TotalSeconds - reserve;
            if (best is not null && clock.Elapsed.TotalSeconds >= deadline)
                break;
            if (_request.MaxTrials is { } max && trialId >= max && best is not null)
                break;

            // Past the first success, only a schema whose last trial fits the time left may run next.
            var left = deadline - clock.Elapsed.TotalSeconds;
            var schemaId = proposer.Propose(id => best is null || !lastSeconds.TryGetValue(id, out var expected) || expected <= left);
            if (schemaId is null)
                break;
            var schema = schemas.First(s => s.Id == schemaId);
            var point = tuners[schemaId].Propose(trialId);
            var trialContext = new MLContext(settings.Seed);
            var featurizer = Featurizer.Build(trialContext, train.Schema, roles, schema.Encoding, withMissingValues, columns.Features)
                ?? throw new InvalidOperationException("No column is a feature: every column is the label or ignored.");

            // Past the first success, a trial that would eat the refit's time stops inside its fit
            // (FitCancellation): one fit can be most of a budget, so stopping between folds is not enough.
            using var budgetCut = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            if (best is not null)
                budgetCut.CancelAfter(TimeSpan.FromSeconds(Math.Max(deadline - clock.Elapsed.TotalSeconds, 0)));

            var started = clock.Elapsed;
            var parameters = Parameters(point, schema.Encoding, encodings.Length > 1);
            // A trial cut by the budget records nothing, so its schema keeps the last finished duration.
            void Took(double seconds) => lastSeconds[schemaId] = seconds;
            IEstimator<ITransformer>? pipeline = null;
            try
            {
                // Built inside the trial: a trainer that cannot even be constructed is a failed trial, not a failed search.
                var featurized = _request.Preprocess is { } preprocess ? preprocess.Append(featurizer) : featurizer;
                pipeline = featurized.Append(schema.Candidate.Build(trialContext, point, columns, settings));
                var score = Validator.Validate(trialContext, train, plan, pipeline, _request.Score, budgetCut.Token);
                var seconds = (clock.Elapsed - started).TotalSeconds;
                Took(seconds);
                var trial = new TrialResult
                {
                    Trainer = schema.Candidate.Name,
                    Parameters = parameters,
                    Metrics = score.Mean,
                    Folds = score.Folds,
                    StandardDeviation = score.StandardDeviation,
                    RuntimeSeconds = seconds,
                };
                trials.Add(trial);
                if (_request.Metric.TryScore(trial, out var value))
                {
                    var loss = _request.Metric.HigherIsBetter ? -value : value;
                    tuners[schemaId].Update(trialId, point, loss, schema.Candidate.InitialCost);
                    proposer.Update(schemaId, loss, schema.Candidate.InitialCost);
                    if (best is null || _request.Metric.IsBetter(value, best.Value.Value))
                        best = (trial, value, score, pipeline, trialContext, seconds / plan.Folds);
                }
                else
                {
                    tuners[schemaId].Update(trialId, point, double.MaxValue, schema.Candidate.InitialCost);
                    proposer.Update(schemaId, double.NaN, schema.Candidate.InitialCost);
                }
                _sink?.Completed(trial, _request.Metric.Name);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                throw;
            }
            catch (OperationCanceledException) when (best is not null)
            {
                break;   // the budget's cut: the trial is dropped, the search ends with what it has
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                var seconds = (clock.Elapsed - started).TotalSeconds;
                Took(seconds);
                var failed = new TrialResult { Trainer = schema.Candidate.Name, Parameters = parameters, RuntimeSeconds = seconds, Error = ex.GetBaseException().Message };
                trials.Add(failed);
                tuners[schemaId].Update(trialId, point, double.MaxValue, schema.Candidate.InitialCost);
                proposer.Update(schemaId, double.NaN, schema.Candidate.InitialCost);
                _sink?.Completed(failed, _request.Metric.Name);
                if (best is null && clock.Elapsed >= _request.Budget)
                    break;
            }
        }

        if (best is not { } winner)
        {
            throw new SearchFailedException(trials,
                $"No trial produced a score in {_request.Budget.TotalSeconds:0} s ({trials.Count} tried): "
                + string.Join("; ", trials.Where(t => t.Error is not null).Select(t => $"{t.Trainer}: {t.Error}").Distinct().Take(5)));
        }

        cancellationToken.ThrowIfCancellationRequested();
        string? fallback = null;
        ITransformer model;
        // The caller's cancellation stops the refit too; the budget does not — the refit was reserved for.
        using var stopRefit = FitCancellation.Register(winner.Context, cancellationToken);
        try
        {
            model = winner.Pipeline.Fit(train);
        }
        catch (Exception ex) when (ex is not OperationCanceledException and not OutOfMemoryException)
        {
            // The pipeline fitted on four fifths of the rows; refit on all of them did not. Its first fold's model stands in.
            fallback = $"Refit on the whole training partition failed ({ex.GetBaseException().Message}); the model is the best trial's first fold.";
            var view = new HoldoutView(train, plan.FoldOf, plan.Validated[0]);
            model = winner.Pipeline.Fit(new MLContext(settings.Seed).Data.FilterRowsByColumn(view, HoldoutView.ColumnName, 0, 0.5));
        }
        return new SearchResult(trials, winner.Trial, winner.Score, model, fallback, plan);
    }

    private static IReadOnlyList<KeyValuePair<string, string>> Parameters(SearchPoint point, CategoricalEncoding encoding, bool withEncoding)
    {
        var values = point.Space.Dimensions.Select(d => new KeyValuePair<string, string>(d.Name, d.Kind == DimensionKind.Choice
            ? point.Choice(d.Name)
            : point[d.Name].ToString("R", CultureInfo.InvariantCulture))).ToList();
        if (withEncoding)
            values.Add(new("CategoricalEncoding", encoding == CategoricalEncoding.OneHot ? "one-hot" : "hash"));
        return values;
    }
}
