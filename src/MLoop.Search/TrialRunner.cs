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
/// <param name="Best">
/// The trial the model is refit from: the highest-scoring (<see cref="TopScoring"/>) — or, when its model is at least
/// <see cref="TrialRunner.LargeModelSize"/>, the smallest model (<see cref="TrainerCandidate.ModelSize"/>) within one
/// standard error of its score.
/// </param>
/// <param name="BestValidation">The best trial's fold scores.</param>
/// <param name="Model">The best pipeline refit on the whole training partition — or, when that refit failed, its first fold's model.</param>
/// <param name="FallbackReason">Why <see cref="Model"/> is a fold's model, or <see langword="null"/>.</param>
/// <param name="Folds">How the partition was validated.</param>
/// <param name="TopScoring">The trial with the highest score — <see cref="Best"/> itself unless a smaller model scored within its standard error.</param>
public sealed record SearchResult(
    IReadOnlyList<TrialResult> Trials,
    TrialResult Best,
    ValidationScore BestValidation,
    ITransformer Model,
    string? FallbackReason,
    FoldPlan Folds,
    TrialResult TopScoring);

/// <summary>No trial of a search produced a score. <see cref="Trials"/> says why each failed.</summary>
public sealed class SearchFailedException(IReadOnlyList<TrialResult> trials, string message) : Exception(message)
{
    /// <summary>The trials run, every one failed.</summary>
    public IReadOnlyList<TrialResult> Trials { get; } = trials;
}

/// <summary>
/// Runs a search: ECI picks the trainer, its cost-frugal tuner picks the
/// settings, the validator scores the pipeline as a mean over folds, and the best is refit on the whole
/// training partition.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item><b>The first trial always finishes</b>, whatever the budget — a search returns a model.</item>
/// <item><b>A large top model gives way to the smallest model within one standard error of its score</b> — the
/// one-standard-error rule of model selection, applied from <see cref="LargeModelSize"/>. Two trials inside that margin cannot be told apart on these rows,
/// and the larger model costs time to fit, save, load and score: measured, a forest of 1,135 trees of 1,510 leaves
/// (a 193 MB model) led one of 27 trees of 252 leaves (1 MB) by 0.006 R², inside its holdout's standard error.
/// Size is read from the trial's settings (<see cref="TrainerCandidate.ModelSize"/>), so the choice depends on
/// the trials alone, never on the clock.</item>
/// <item><b>The refit is budgeted</b>: the search stops starting trials once the time left is what refitting
/// the chosen trial is expected to take (its fit on one fold's training rows × the share those rows are of the
/// partition — parts ⁄ (parts − 1), for folds and holdout alike), and a trial still running at that
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
    /// <summary>
    /// The model size (<see cref="TrainerCandidate.ModelSize"/>, trees × leaves) from which the search keeps a smaller
    /// model within one standard error of the top score. Below it a model is a few megabytes at most and costs
    /// nothing worth a point of score; above it — a hundred thousand leaves — it is tens to hundreds of megabytes,
    /// seconds to save and load, and its save spills past the time budget. Measured on twelve sets: where the top
    /// model was smaller, keeping a smaller one still lost up to 0.03 of score for no saving.
    /// </summary>
    public const double LargeModelSize = 100_000;

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

        var schemas = _request.Candidates.Select(c => (Id: c.Name, Candidate: c)).ToList();
        var proposer = new EciSchemaProposer([.. schemas.Select(s => (s.Id, s.Candidate.InitialCost))], settings.Seed);
        var tuners = schemas.ToDictionary(s => s.Id, s => new CostFrugalTuner(s.Candidate.Space, settings.Seed), StringComparer.Ordinal);

        var trials = new List<TrialResult>();
        var lastSeconds = new Dictionary<string, double>(StringComparer.Ordinal);
        var scored = new List<Scored>();
        Scored? best = null;     // the highest score so far

        for (var trialId = 0; ; trialId++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var reserve = best is null ? 0 : Choose(scored, best).FitSeconds * plan.Parts / Math.Max(plan.Parts - 1, 1);
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
            // The trial's pipeline, made in a given context — the trial's own, and again for the refit, whose
            // context must not be one a budget cut has stopped (FitCancellation marks the context, not the fit).
            IEstimator<ITransformer> PipelineIn(MLContext context)
            {
                var featurizer = Featurizer.Build(context, train.Schema, roles, withMissingValues, columns.Features)
                    ?? throw new InvalidOperationException("No column is a feature: every column is the label or ignored.");
                var featurized = _request.Preprocess is { } preprocess ? preprocess.Append(featurizer) : featurizer;
                return featurized.Append(schema.Candidate.Build(context, point, columns, settings));
            }

            // Past the first success, a trial that would eat the refit's time stops inside its fit
            // (FitCancellation): one fit can be most of a budget, so stopping between folds is not enough.
            using var budgetCut = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            if (best is not null)
                budgetCut.CancelAfter(TimeSpan.FromSeconds(Math.Max(deadline - clock.Elapsed.TotalSeconds, 0)));

            var started = clock.Elapsed;
            var parameters = Parameters(point);
            // A trial cut by the budget records nothing, so its schema keeps the last finished duration.
            void Took(double seconds) => lastSeconds[schemaId] = seconds;
            IEstimator<ITransformer>? pipeline = null;
            try
            {
                // Built inside the trial: a trainer that cannot even be constructed is a failed trial, not a failed search.
                pipeline = PipelineIn(trialContext);
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
                    var entry = new Scored(trial, value, score, PipelineIn, seconds / plan.Folds, TrainerCandidate.ModelSize(point));
                    scored.Add(entry);
                    if (best is null || _request.Metric.IsBetter(value, best.Value))
                        best = entry;
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

        if (best is not { } top)
        {
            throw new SearchFailedException(trials,
                $"No trial produced a score in {_request.Budget.TotalSeconds:0} s ({trials.Count} tried): "
                + string.Join("; ", trials.Where(t => t.Error is not null).Select(t => $"{t.Trainer}: {t.Error}").Distinct().Take(5)));
        }

        cancellationToken.ThrowIfCancellationRequested();
        var winner = Choose(scored, top);
        string? fallback = null;
        ITransformer model;
        // The refit fits in a context of its own: the winner's trial context may carry the budget's cut, set as
        // the trial finished — measured on a CI runner, the refit then died with OperationCanceledException.
        // The caller's cancellation stops the refit too; the budget does not — the refit was reserved for.
        var refitContext = new MLContext(settings.Seed);
        using var stopRefit = FitCancellation.Register(refitContext, cancellationToken);
        try
        {
            model = winner.Pipeline(refitContext).Fit(train);
        }
        catch (Exception ex) when (ex is not OperationCanceledException and not OutOfMemoryException)
        {
            // The pipeline fitted on four fifths of the rows; refit on all of them did not. Its first fold's model stands in.
            fallback = $"Refit on the whole training partition failed ({ex.GetBaseException().Message}); the model is the best trial's first fold.";
            var view = new HoldoutView(train, plan.FoldOf, plan.Validated[0]);
            var fallbackContext = new MLContext(settings.Seed);
            model = winner.Pipeline(fallbackContext).Fit(fallbackContext.Data.FilterRowsByColumn(view, HoldoutView.ColumnName, 0, 0.5));
        }
        return new SearchResult(trials, winner.Trial, winner.Score, model, fallback, plan, top.Trial);
    }

    private sealed record Scored(
        TrialResult Trial, double Value, ValidationScore Score, Func<MLContext, IEstimator<ITransformer>> Pipeline, double FitSeconds, double Size);

    private Scored Choose(List<Scored> scored, Scored top)
    {
        var margin = top.Score.StandardError is { } errors && errors.TryGetValue(_request.Metric.Name, out var se) && double.IsFinite(se) ? se : 0;
        return scored[Choose([.. scored.Select(t => (t.Value, t.Size))], scored.IndexOf(top), margin, _request.Metric)];
    }

    /// <summary>
    /// The trial kept: when the top-scoring trial's model is at least <see cref="LargeModelSize"/>, the smallest model
    /// within <paramref name="margin"/> of its score — the better score, then the earlier trial, between two of one
    /// size; otherwise the top trial itself.
    /// </summary>
    internal static int Choose(IReadOnlyList<(double Value, double Size)> trials, int top, double margin, IMetricPolicy metric)
    {
        if (trials[top].Size < LargeModelSize)
            return top;
        var floor = metric.HigherIsBetter ? trials[top].Value - margin : trials[top].Value + margin;
        var chosen = top;
        for (var i = 0; i < trials.Count; i++)
        {
            var (value, size) = trials[i];
            var inside = metric.HigherIsBetter ? value >= floor : value <= floor;
            if (inside && (size < trials[chosen].Size || (size == trials[chosen].Size && metric.IsBetter(value, trials[chosen].Value))))
                chosen = i;
        }
        return chosen;
    }

    private static IReadOnlyList<KeyValuePair<string, string>> Parameters(SearchPoint point)
        => [.. point.Space.Dimensions.Select(d => new KeyValuePair<string, string>(d.Name, d.Kind == DimensionKind.Choice
            ? point.Choice(d.Name)
            : point[d.Name].ToString("R", CultureInfo.InvariantCulture)))];
}
