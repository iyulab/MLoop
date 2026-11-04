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
/// <param name="SampleLargeTables">
/// Probe a table of <see cref="SearchSample.ProbeRows"/> rows or more, and search a sample of it when the first trial on all the rows
/// would take more than a tenth of the budget (<see cref="SearchSample"/>). Off, the search is on all the rows whatever the first trial costs.
/// </param>
/// <param name="KeepTopOutOfFold">
/// Also keep the answers of the best this-many trials, whatever their distance from the top, and return them with the
/// pipelines that made them (<see cref="SearchResult.Retained"/>) — what an ensemble of the best trials is chosen and weighted
/// on. A regression's answers are read too. Needs <see cref="KeepOutOfFold"/>. Costs memory in proportion to the rows and
/// this number, and nothing when 0 (the default), which keeps only what a calibration of the kept trial needs.
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
    bool KeepOutOfFold = false,
    int KeepTopOutOfFold = 0,
    bool SampleLargeTables = true);

/// <summary>A trial whose answers on its validated parts were kept beyond the winner's (<see cref="SearchRequest.KeepTopOutOfFold"/>).</summary>
/// <param name="Trial">The trial.</param>
/// <param name="OutOfFold">Its answers on every row of the partition once, each by the model fitted without that row's fold.</param>
/// <param name="Pipeline">Builds the trial's pipeline in a context — fit it on the whole partition to get the member's final model.</param>
/// <param name="FitSeconds">
/// About how long fitting <paramref name="Pipeline"/> on the whole partition takes: a fold's fit and scoring scaled up by the
/// share of the partition a fold leaves out — the estimate the search itself reserves for the refit of its best trial. 0 when unknown.
/// </param>
public sealed record RetainedTrial(
    TrialResult Trial, OutOfFoldScores OutOfFold, Func<MLContext, IEstimator<ITransformer>> Pipeline, double FitSeconds = 0);

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
/// and the pipeline is a classifier (or a regression, when <see cref="SearchRequest.KeepTopOutOfFold"/> asked for those too);
/// <see langword="null"/> otherwise.
/// </param>
/// <param name="Retained">
/// The best <see cref="SearchRequest.KeepTopOutOfFold"/> trials with their answers, best first; <see langword="null"/> when none were asked for.
/// </param>
/// <param name="Sampling">How many rows the search ran on and the model was fitted on, when the table was probed (<see cref="SearchSample"/>); null below it.</param>
public sealed record SearchResult(
    IReadOnlyList<TrialResult> Trials,
    TrialResult Best,
    ValidationScore BestValidation,
    ITransformer Model,
    string? FallbackReason,
    FoldPlan Folds,
    TrialResult TopScoring,
    OutOfFoldScores? OutOfFold = null,
    IReadOnlyList<RetainedTrial>? Retained = null,
    SamplingOutcome? Sampling = null);

/// <summary>What probing a large table decided (<see cref="SearchSample"/>).</summary>
/// <param name="Rows">Rows of the training partition.</param>
/// <param name="SearchRows">Rows the trials ran on — <paramref name="Rows"/> unless the table was sampled.</param>
/// <param name="RefitRows">Rows the model was fitted on: <paramref name="Rows"/>, or the sample when a refit on all of them would not fit the time budget.</param>
/// <param name="ProbeSeconds">How long the probe took, the search's clock included.</param>
public sealed record SamplingOutcome(int Rows, int SearchRows, int RefitRows, double ProbeSeconds)
{
    /// <summary>Whether the trials ran on fewer rows than the table has.</summary>
    public bool Sampled => SearchRows < Rows;
}

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
        ArgumentOutOfRangeException.ThrowIfNegative(request.KeepTopOutOfFold);
        if (request.KeepTopOutOfFold > 0 && !request.KeepOutOfFold)
            throw new ArgumentException("Keeping the best trials' answers needs the answers kept (KeepOutOfFold).", nameof(request));
        _sink = sink;
    }

    /// <summary>Searches <paramref name="train"/> (the training partition) for <paramref name="roles"/>' label.</summary>
    public SearchResult Run(IDataView train, ColumnRoles roles, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(train);
        ArgumentNullException.ThrowIfNull(roles);
        var clock = Stopwatch.StartNew();
        var settings = _request.Settings;
        var keys = FoldPlan.LabelKeys(train, roles.Label);
        // Every fold of every trial reads the training rows several times, and from a file each read parses them again — measured, a
        // linear or SDCA trial of a few seconds took a fifth of that once the rows were in memory, and sixteen public tables made 28 %
        // more trials in the same minute. Kept in memory when they fit (about four bytes a numeric cell, a kilobyte a text one); a table past the limit is
        // read from its source as before.
        var textColumns = roles.Of(ColumnRole.Text).Count;
        var trainInMemory = KeepsRowsInMemory(keys.Length, train.Schema.Count, textColumns);
        if (trainInMemory)
            train = RowCache.Of(new MLContext(settings.Seed), train);
        var columns = new TrainerColumns(roles.Label, "Features", _request.Weight);
        var withMissingValues = Featurizer.ColumnsWithMissingValues(train, roles);

        var schemas = _request.Candidates.Select(c => (Id: c.Name, Candidate: c)).ToList();
        // A trainer that fits a model per class costs the more the more classes there are (TrainerCandidate.CostFor): the proposer
        // starts from that, so a label of a hundred classes does not open with a trainer that takes seconds a fold on a few thousand rows.
        var classes = _request.Stratify ? keys.Distinct(StringComparer.Ordinal).Count() : 1;
        var proposer = new EciSchemaProposer([.. schemas.Select(s => (s.Id, s.Candidate.CostFor(classes)))], settings.Seed);
        var tuners = schemas.ToDictionary(s => s.Id, s => new CostFrugalTuner(s.Candidate.Space, settings.Seed, startingPoints: [.. s.Candidate.StartingPoints.Select(s.Candidate.Space.PointOf)]), StringComparer.Ordinal);

        // A large table is probed: the first trial on a sample says what one trial on all the rows would cost, and when that is
        // more than the search can spend, the search runs on a sample (SearchSample). Below the threshold none of this runs.
        var searchTrain = train;
        var searchKeys = (IReadOnlyList<string>)keys;
        SamplingOutcome? sampling = null;
        if (_request.SampleLargeTables && keys.Length >= SearchSample.ProbeRows)
        {
            var probe = ProbeSeconds(train, roles, keys, schemas, classes, withMissingValues, columns, trainInMemory, clock, cancellationToken);
            var probeSeconds = probe?.Seconds;
            // A probe cut short says even the smallest sample's first trial is over the allowance: the search runs on that sample.
            var searchRows = probe is not { } timed ? keys.Length
                : timed.Cut ? Math.Min(SearchSample.ProbeSample, keys.Length)
                : SearchSample.RowsToSearch(keys.Length, _request.Budget.TotalSeconds, timed.Seconds, SearchSample.ProbeSample);
            if (searchRows < keys.Length)
            {
                var keep = SearchSample.Stratified(keys, _request.Stratify, searchRows, settings.Seed);
                searchTrain = SearchSample.Rows(new MLContext(settings.Seed), train, keep,
                    cache: trainInMemory ? searchRows * SearchSample.CopyBelow <= keys.Length : KeepsRowsInMemory(searchRows, train.Schema.Count, textColumns));
                searchKeys = [.. keys.Where((_, i) => keep[i])];
            }
            sampling = new SamplingOutcome(keys.Length, searchRows, keys.Length, probeSeconds ?? 0);
        }
        var plan = FoldPlan.ForKeys(searchKeys, _request.Stratify, settings.Seed);
        // What a refit of the winner costs against the fit the trial measured: on all the rows, the sample's fit scaled by the rows.
        var scale = (double)keys.Length / searchKeys.Count;

        var trials = new List<TrialResult>();
        var lastSeconds = new Dictionary<string, double>(StringComparer.Ordinal);
        var failuresWithoutScore = new Dictionary<string, int>(StringComparer.Ordinal);
        var retired = new HashSet<string>(StringComparer.Ordinal);
        var scored = new List<Scored>();
        Scored? best = null;     // the highest score so far

        for (var trialId = 0; ; trialId++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var reserve = best is null ? 0 : RefitReserve(Choose(scored, best).FitSeconds * plan.Parts / Math.Max(plan.Parts - 1, 1), scale);
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
            IEstimator<ITransformer> PipelineIn(MLContext context) => PipelineFor(context, train.Schema, roles, withMissingValues, columns, schema.Candidate, point, plan.FoldOf.Count);

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
                if (readable && OutOfFoldScores.Read(part, roles.Label, _request.Weight, outOfFold, regression: _request.KeepTopOutOfFold > 0) is { } rows)
                    outOfFold = rows;
                else
                    (readable, outOfFold) = (false, null);
            }
            try
            {
                // Built inside the trial: a trainer that cannot even be constructed is a failed trial, not a failed search.
                pipeline = PipelineIn(trialContext);
                var score = Validator.Validate(trialContext, searchTrain, plan, pipeline, _request.Score, budgetCut.Token,
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
                    FoldMetrics = score.IsHoldout ? null : score.FoldMetrics,
                    RuntimeSeconds = seconds,
                };
                trials.Add(trial);
                if (_request.Metric.TryScore(trial, out var value))
                {
                    var loss = _request.Metric.HigherIsBetter ? -value : value;
                    tuners[schemaId].Update(trialId, point, loss, schema.Candidate.CostFor(classes));
                    proposer.Update(schemaId, loss, schema.Candidate.CostFor(classes));
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
                    tuners[schemaId].Update(trialId, point, double.MaxValue, schema.Candidate.CostFor(classes));
                    proposer.Update(schemaId, double.NaN, schema.Candidate.CostFor(classes));
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
                tuners[schemaId].Update(trialId, point, double.MaxValue, schema.Candidate.CostFor(classes));
                proposer.Update(schemaId, double.NaN, schema.Candidate.CostFor(classes));
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
        // The model is fitted on all the rows unless the search ran on a sample and a refit on all of them would take more than
        // the share of the budget SearchSample allows — then on the sample, which is what the time bought.
        var refitRows = scale > 1 && !SearchSample.RefitsOnAllRows(
            keys.Length, searchKeys.Count, winner.FitSeconds * plan.Parts / Math.Max(plan.Parts - 1, 1), _request.Budget.TotalSeconds)
            ? searchKeys.Count : keys.Length;
        string? fallback = null;
        ITransformer model;
        // The refit fits in a context of its own: the winner's trial context may carry the budget's cut, set as
        // the trial finished — measured on a CI runner, the refit then died with OperationCanceledException.
        // The caller's cancellation stops the refit too; the budget does not — the refit was reserved for.
        var refitContext = new MLContext(settings.Seed);
        using var stopRefit = FitCancellation.Register(refitContext, cancellationToken);
        try
        {
            model = winner.Pipeline(refitContext).Fit(refitRows == keys.Length ? train : searchTrain);
        }
        catch (Exception ex) when (ex is not OperationCanceledException and not OutOfMemoryException)
        {
            // The pipeline fitted on four fifths of the rows; refit on all of them did not. Its first fold's model stands in.
            fallback = $"Refit on the whole training partition failed ({ex.GetBaseException().Message}); the model is the best trial's first fold.";
            var view = new HoldoutView(searchTrain, plan.FoldOf, plan.Validated[0]);
            var fallbackContext = new MLContext(settings.Seed);
            model = winner.Pipeline(fallbackContext).Fit(fallbackContext.Data.FilterRowsByColumn(view, HoldoutView.ColumnName, 0, 0.5));
        }
        IReadOnlyList<RetainedTrial>? retained = _request.KeepTopOutOfFold > 0
            ? [.. BestFirst(scored).Where(e => e.OutOfFold is not null).Take(_request.KeepTopOutOfFold)
                .Select(e => new RetainedTrial(e.Trial, e.OutOfFold!, e.Pipeline, e.FitSeconds * plan.Parts / Math.Max(plan.Parts - 1, 1) * scale))]
            : null;
        return new SearchResult(trials, winner.Trial, winner.Score, model, fallback, plan, top.Trial, winner.OutOfFold, retained,
            sampling is null ? null : sampling with { RefitRows = refitRows });
    }

    // The refit the search reserves time for: on all the rows (the sample's fit scaled by the rows) when that is within the share of
    // the budget SearchSample allows, else the sample's own.
    private double RefitReserve(double sampleRefitSeconds, double scale)
        => scale > 1 && sampleRefitSeconds * scale <= SearchSample.RefitShare * _request.Budget.TotalSeconds ? sampleRefitSeconds * scale : sampleRefitSeconds;

    private IEstimator<ITransformer> PipelineFor(
        MLContext context, DataViewSchema schema, ColumnRoles roles, IReadOnlySet<string> withMissingValues, TrainerColumns columns,
        TrainerCandidate candidate, SearchPoint point, int rows)
    {
        var featurizer = Featurizer.Build(context, schema, roles, withMissingValues, columns.Features)
            ?? throw new InvalidOperationException("No column is a feature: every column is the label or ignored.");
        var featurized = _request.Preprocess is { } preprocess ? preprocess.Append(featurizer) : featurizer;
        return featurized.Append(candidate.Build(context, TrainerCandidate.FittedToRows(point, rows), columns, _request.Settings));
    }

    // The search's own first trial — the cheapest schema at its starting setting — on a stratified sample of SearchSample.ProbeSample rows,
    // timed; null when the trial fails (the search then runs on all the rows and records the failure itself). Made from a twin of the
    // proposer and the tuner, so the search proposes exactly what it would have; the probe's time is on the search's clock. A probe
    // is stopped at SearchSample.ProbeCutShare of the allowance for a first trial — a table so wide or a trainer so slow that 10,000 rows
    // already pass the allowance would otherwise spend the budget on measuring it — and reports that it was cut.
    private (double Seconds, bool Cut)? ProbeSeconds(
        IDataView train, ColumnRoles roles, string[] keys, List<(string Id, TrainerCandidate Candidate)> schemas, int classes,
        IReadOnlySet<string> withMissingValues, TrainerColumns columns, bool trainInMemory, Stopwatch clock, CancellationToken cancellationToken)
    {
        var settings = _request.Settings;
        var keep = SearchSample.Stratified(keys, _request.Stratify, SearchSample.ProbeSample, settings.Seed);
        var probeKeys = keys.Where((_, i) => keep[i]).ToArray();
        var started = clock.Elapsed;
        try
        {
            var data = SearchSample.Rows(new MLContext(settings.Seed), train, keep, cache: !trainInMemory);
            var probePlan = FoldPlan.ForKeys(probeKeys, _request.Stratify, settings.Seed);
            var first = new EciSchemaProposer([.. schemas.Select(s => (s.Id, s.Candidate.CostFor(classes)))], settings.Seed).Propose();
            var schema = schemas.First(s => s.Id == first);
            var point = new CostFrugalTuner(schema.Candidate.Space, settings.Seed, startingPoints: [.. schema.Candidate.StartingPoints.Select(schema.Candidate.Space.PointOf)]).Propose(0);
            var context = new MLContext(settings.Seed);
            var pipeline = PipelineFor(context, train.Schema, roles, withMissingValues, columns, schema.Candidate, point, probeKeys.Length);
            var fitStarted = clock.Elapsed;
            var cutAfter = SearchSample.ProbeCutShare * SearchSample.FirstTrialShare * _request.Budget.TotalSeconds;
            using var cut = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            cut.CancelAfter(TimeSpan.FromSeconds(cutAfter));
            try
            {
                Validator.Validate(context, data, probePlan, pipeline, _request.Score, cut.Token);
            }
            catch (Exception ex) when (ex is not OutOfMemoryException && !cancellationToken.IsCancellationRequested && cut.IsCancellationRequested)
            {
                return ((clock.Elapsed - fitStarted).TotalSeconds, true);   // stopped inside its fit (or failed because it was)
            }
            var seconds = (clock.Elapsed - fitStarted).TotalSeconds;
            return (seconds, seconds >= cutAfter);
        }
        catch (Exception ex) when (ex is not OperationCanceledException and not OutOfMemoryException)
        {
            return null;
        }
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
    // A request for the best few trials' answers (KeepTopOutOfFold) spares those whatever their distance from the top.
    private void DropDistantOutOfFold(List<Scored> scored, Scored top)
    {
        var keeps = KeepsOutOfFold([.. scored.Select(e => e.Value)], scored.IndexOf(top), 2 * Margin(top), _request.Metric.HigherIsBetter, _request.KeepTopOutOfFold);
        for (var i = 0; i < scored.Count; i++)
        {
            if (!keeps[i])
                scored[i].OutOfFold = null;
        }
    }

    /// <summary>
    /// Which trials keep their answers: the top, any within <paramref name="margin"/> of its score, and the best
    /// <paramref name="keepTop"/> by score whatever their distance (the earlier trial between two of one score).
    /// </summary>
    internal static bool[] KeepsOutOfFold(IReadOnlyList<double> values, int top, double margin, bool higherIsBetter, int keepTop)
    {
        var floor = higherIsBetter ? values[top] - margin : values[top] + margin;
        var keeps = new bool[values.Count];
        for (var i = 0; i < values.Count; i++)
            keeps[i] = i == top || (higherIsBetter ? values[i] >= floor : values[i] <= floor);
        if (keepTop > 0)
        {
            var order = Enumerable.Range(0, values.Count);
            foreach (var i in (higherIsBetter ? order.OrderByDescending(i => values[i]) : order.OrderBy(i => values[i])).Take(keepTop))
                keeps[i] = true;
        }
        return keeps;
    }

    // Best score first; between two of one score the earlier trial (the sort is stable).
    private IEnumerable<Scored> BestFirst(List<Scored> scored)
        => _request.Metric.HigherIsBetter ? scored.OrderByDescending(e => e.Value) : scored.OrderBy(e => e.Value);

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
