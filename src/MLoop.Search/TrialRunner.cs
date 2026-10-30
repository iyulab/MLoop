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
/// <param name="KeepOutOfFold">
/// Keep a classifier's answers on its validated parts (<see cref="OutOfFoldScores"/>), so the kept trial's can be
/// returned (<see cref="SearchResult.OutOfFold"/>) — what a calibration of the refit model is fitted on. Costs one more
/// pass over each validated part; trials whose score falls well outside the top's standard error drop theirs.
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
    IEstimator<ITransformer>? Preprocess = null,
    bool KeepOutOfFold = false);

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
/// <param name="OutOfFold">
/// <see cref="Best"/>'s answers on its validated parts, when the request kept them (<see cref="SearchRequest.KeepOutOfFold"/>)
/// and the pipeline is a classifier; <see langword="null"/> otherwise.
/// </param>
public sealed record SearchResult(
    IReadOnlyList<TrialResult> Trials,
    TrialResult Best,
    ValidationScore BestValidation,
    ITransformer Model,
    string? FallbackReason,
    FoldPlan Folds,
    TrialResult TopScoring,
    OutOfFoldScores? OutOfFold = null);

/// <summary>No trial of a search produced a score. <see cref="Trials"/> says why each failed.</summary>
public sealed class SearchFailedException(IReadOnlyList<TrialResult> trials, string message, bool allTrainersRetired = false) : Exception(message)
{
    /// <summary>The trials run, every one failed.</summary>
    public IReadOnlyList<TrialResult> Trials { get; } = trials;

    /// <summary>
    /// Whether the search stopped because every trainer had failed <see cref="TrialRunner.FailuresToRetire"/> times in a row —
    /// before its time budget ran out, so more time would not have changed it. False when the budget ended with nothing scored.
    /// </summary>
    public bool AllTrainersRetired { get; } = allTrainersRetired;
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
    /// model within one standard error of the top score. A gradient-boosted model of ten million leaves is about
    /// 200 MB on disk and takes seconds to save, load and score, so above it a point of score inside the margin is not
    /// worth the model. Below it the rule cost score: on sixteen public tables, keeping the top-scoring trial instead
    /// of the smallest model within one standard error was better by up to 0.0066 RMSE and 0.0063 AUC (twice its standard
    /// error on two tables) and worse on none the rule touched, at 60 and 240 seconds; the largest top model in those
    /// runs was 6.7 million leaves (150 MB, 6.7 s to score 4,100 rows). The rule used to start at a hundred thousand
    /// leaves, which an ordinary model of a few hundred trees reaches.
    /// </summary>
    public const double LargeModelSize = 10_000_000;

    /// <summary>
    /// How many failures in a row, without one scored trial, retire a trainer from the search. A trainer whose first few trials —
    /// the cheapest settings it has — all fail (a native library that is not there, a column it cannot read) is not going to
    /// succeed on a later setting, and each further try spends budget a trainer that works could use. A trainer that has
    /// scored is never retired: its failures belong to the settings that failed, which the tuner already steers away from.
    /// </summary>
    public const int FailuresToRetire = 3;

    /// <summary>Cells up to which the search keeps the training rows in memory (50 million: a few hundred megabytes).</summary>
    internal const long CacheCellLimit = 50_000_000;

    /// <summary>What a text cell weighs against a number's four bytes: a kilobyte of language, so a corpus of long documents is not kept whole.</summary>
    internal const int TextCellWeight = 250;

    internal static bool KeepsRowsInMemory(long rows, int columns, int textColumns)
        => rows * (columns - textColumns + (long)TextCellWeight * textColumns) <= CacheCellLimit;

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
        // Every fold of every trial reads the training rows several times, and from a file each read parses them again — measured, a
        // linear or SDCA trial of a few seconds took a fifth of that once the rows were in memory, and sixteen public tables made 28 %
        // more trials in the same minute. Kept in memory when they fit (about four bytes a numeric cell, a kilobyte a text one); a table past the limit is
        // read from its source as before.
        if (KeepsRowsInMemory(plan.FoldOf.Count, train.Schema.Count, roles.Of(ColumnRole.Text).Count))
            train = new MLContext(settings.Seed).Data.Cache(train);
        var columns = new TrainerColumns(roles.Label, "Features", _request.Weight);
        var withMissingValues = Featurizer.ColumnsWithMissingValues(train, roles);

        var schemas = _request.Candidates.Select(c => (Id: c.Name, Candidate: c)).ToList();
        var proposer = new EciSchemaProposer([.. schemas.Select(s => (s.Id, s.Candidate.InitialCost))], settings.Seed);
        var tuners = schemas.ToDictionary(s => s.Id, s => new CostFrugalTuner(s.Candidate.Space, settings.Seed, startingPoints: [.. s.Candidate.StartingPoints.Select(s.Candidate.Space.PointOf)]), StringComparer.Ordinal);

        var trials = new List<TrialResult>();
        var lastSeconds = new Dictionary<string, double>(StringComparer.Ordinal);
        var failuresWithoutScore = new Dictionary<string, int>(StringComparer.Ordinal);
        var retired = new HashSet<string>(StringComparer.Ordinal);
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
            var schemaId = proposer.Propose(id => !retired.Contains(id) && (best is null || !lastSeconds.TryGetValue(id, out var expected) || expected <= left));
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
            // The trial's answers on its validated parts, when the request keeps them; a part that is not binary ends it.
            OutOfFoldScores? outOfFold = null;
            var readable = _request.KeepOutOfFold;
            void Keep(IDataView part)
            {
                if (readable && OutOfFoldScores.Read(part, roles.Label, _request.Weight, outOfFold) is { } rows)
                    outOfFold = rows;
                else
                    (readable, outOfFold) = (false, null);
            }
            try
            {
                // Built inside the trial: a trainer that cannot even be constructed is a failed trial, not a failed search.
                pipeline = PipelineIn(trialContext);
                var score = Validator.Validate(trialContext, train, plan, pipeline, _request.Score, budgetCut.Token,
                    _request.KeepOutOfFold ? Keep : null);
                var seconds = (clock.Elapsed - started).TotalSeconds;
                Took(seconds);
                var trial = new TrialResult
                {
                    Trainer = schema.Candidate.Name,
                    Parameters = parameters,
                    Metrics = score.Mean,
                    Folds = score.Folds,
                    StandardDeviation = score.StandardDeviation,
                    StandardError = score.StandardError,
                    RuntimeSeconds = seconds,
                };
                trials.Add(trial);
                if (_request.Metric.TryScore(trial, out var value))
                {
                    var loss = _request.Metric.HigherIsBetter ? -value : value;
                    tuners[schemaId].Update(trialId, point, loss, schema.Candidate.InitialCost);
                    proposer.Update(schemaId, loss, schema.Candidate.InitialCost);
                    var entry = new Scored(trial, value, score, PipelineIn, seconds / plan.Folds, TrainerCandidate.ModelSize(point))
                    {
                        OutOfFold = outOfFold,
                    };
                    scored.Add(entry);
                    if (best is null || _request.Metric.IsBetter(value, best.Value))
                        best = entry;
                    DropDistantOutOfFold(scored, best);
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
                // A fit the budget's cut stopped can surface as an exception other than OperationCanceledException
                // (a trainer that wraps or loses the cancellation); that is the cut, not a trainer that cannot fit.
                if (best is not null && budgetCut.IsCancellationRequested)
                    break;

                var seconds = (clock.Elapsed - started).TotalSeconds;
                Took(seconds);
                var failed = new TrialResult { Trainer = schema.Candidate.Name, Parameters = parameters, RuntimeSeconds = seconds, Error = ex.GetBaseException().Message };
                trials.Add(failed);
                tuners[schemaId].Update(trialId, point, double.MaxValue, schema.Candidate.InitialCost);
                proposer.Update(schemaId, double.NaN, schema.Candidate.InitialCost);
                _sink?.Completed(failed, _request.Metric.Name);
                // A trainer that has not scored once and keeps failing is retired; one that has scored is not (see FailuresToRetire).
                if (!scored.Any(t => t.Trial.Trainer == schema.Candidate.Name)
                    && (failuresWithoutScore[schemaId] = failuresWithoutScore.GetValueOrDefault(schemaId) + 1) >= FailuresToRetire)
                    retired.Add(schemaId);
                if (best is null && clock.Elapsed >= _request.Budget)
                    break;
            }
        }

        if (best is not { } top)
        {
            var why = string.Join("; ", trials.Where(t => t.Error is not null).Select(t => $"{t.Trainer}: {t.Error}").Distinct().Take(5));
            var allRetired = retired.Count == schemas.Count;
            throw new SearchFailedException(trials, allRetired
                ? $"No trial produced a score: every trainer failed {FailuresToRetire} times in a row and was retired ({trials.Count} tried in {clock.Elapsed.TotalSeconds:0} s of {_request.Budget.TotalSeconds:0} s): {why}"
                : $"No trial produced a score in {_request.Budget.TotalSeconds:0} s ({trials.Count} tried): {why}", allRetired);
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
        return new SearchResult(trials, winner.Trial, winner.Score, model, fallback, plan, top.Trial, winner.OutOfFold);
    }

    private sealed record Scored(
        TrialResult Trial, double Value, ValidationScore Score, Func<MLContext, IEstimator<ITransformer>> Pipeline, double FitSeconds, double Size)
    {
        /// <summary>The trial's answers on its validated parts, while it could still be kept (<see cref="DropDistantOutOfFold"/>).</summary>
        public OutOfFoldScores? OutOfFold { get; set; }
    }

    // Out-of-fold answers are a few numbers per row per trial. Only a trial that Choose could keep needs them — the
    // top, or one within its standard error — so a trial further than twice that from the top drops its own. The
    // margin is read from the scores alone, so which trials keep theirs is as repeatable as the search.
    private void DropDistantOutOfFold(List<Scored> scored, Scored top)
    {
        var margin = 2 * Margin(top);
        var floor = _request.Metric.HigherIsBetter ? top.Value - margin : top.Value + margin;
        foreach (var entry in scored)
        {
            if (entry.OutOfFold is not null && !ReferenceEquals(entry, top)
                && (_request.Metric.HigherIsBetter ? entry.Value < floor : entry.Value > floor))
                entry.OutOfFold = null;
        }
    }

    private double Margin(Scored top)
        => top.Score.StandardError is { } errors && errors.TryGetValue(_request.Metric.Name, out var se) && double.IsFinite(se) ? se : 0;

    private Scored Choose(List<Scored> scored, Scored top)
    {
        var margin = Margin(top);
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
