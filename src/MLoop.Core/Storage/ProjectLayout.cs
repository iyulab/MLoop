namespace MLoop.Core.Storage;

/// <summary>
/// The names at the root of a project: the directories a user puts data in and the product writes
/// results to, and what the files inside them are called. The sibling of
/// <see cref="ExperimentLayout"/>, which owns the names <i>inside</i> a model directory.
/// </summary>
/// <remarks>
/// <para>Two of these names were held apart from each other. <c>"datasets"</c> had a private
/// constant in one class and seven literals elsewhere, including in the API, which cannot see that
/// class. <c>"predictions"</c> had no constant at all.</para>
/// <para>The prediction file names are the part that was actually broken. Three commands write into
/// that directory under three conventions — a table of rows, a set of per-image detections, a
/// forecast — and the one place that reads it back looked for the first convention only. A project
/// whose model does object detection or forecasting therefore reported "no predictions" in
/// <c>mloop status</c> no matter how often it had predicted. The documented convention was a fourth
/// version, from before models were named: <c>predictions/predictions-TIMESTAMP.csv</c>, with no
/// model prefix at all.</para>
/// <para>So the file name is not exposed as a string to interpolate. A writer asks for the name of
/// the file it is about to write and a reader asks for the patterns that find them, and both come
/// from <see cref="PredictionKinds"/> — adding a fourth kind is one entry, and a reader that has
/// not been taught about it is not possible.</para>
/// </remarks>
public static class ProjectLayout
{
    /// <summary>Where the user's data goes.</summary>
    public const string DatasetsDirectory = "datasets";

    /// <summary>Where prediction output goes.</summary>
    public const string PredictionsDirectory = "predictions";

    /// <summary>The training file the conventions look for first; the only mandatory one.</summary>
    public const string TrainFileName = "train.csv";

    /// <summary>Held-out data for <c>evaluate</c>.</summary>
    public const string TestFileName = "test.csv";

    /// <summary>An explicit validation split, when the user supplies one instead of letting MLoop split.</summary>
    public const string ValidationFileName = "validation.csv";

    /// <summary>The rows <c>predict</c> scores when none are named on the command line.</summary>
    public const string PredictFileName = "predict.csv";

    /// <summary>
    /// What a command writes into <see cref="PredictionsDirectory"/>. The name is part of the
    /// convention, not an implementation detail of the command that writes it — <c>status</c> finds
    /// a model's most recent output by looking for exactly these.
    /// </summary>
    public enum PredictionKind
    {
        /// <summary>Scored rows, one per input row.</summary>
        Rows,

        /// <summary>Per-image detections (label, score, box) for object detection.</summary>
        Detections,

        /// <summary>A horizon of future values.</summary>
        Forecast,
    }

    /// <summary>Every kind, with the two things that make its file name.</summary>
    private static readonly IReadOnlyDictionary<PredictionKind, (string Infix, string Extension)> Conventions =
        new Dictionary<PredictionKind, (string, string)>
        {
            [PredictionKind.Rows] = ("predictions", ".csv"),
            [PredictionKind.Detections] = ("detections", ".json"),
            [PredictionKind.Forecast] = ("forecast", ".csv"),
        };

    /// <summary>The kinds, for a caller that needs to cover all of them.</summary>
    public static IEnumerable<PredictionKind> PredictionKinds => Conventions.Keys;

    /// <summary>
    /// The file name for one prediction run: <c>{model}-{kind}-{timestamp}{ext}</c>.
    /// </summary>
    /// <remarks>
    /// The stamp is local, not UTC: this is a name a person scans a directory listing for, and
    /// someone looking for the predictions they ran after lunch should not have to convert. It is
    /// the same rule the CLI's display authority states for file names a user goes looking for.
    /// </remarks>
    public static string PredictionFileName(string modelName, PredictionKind kind, DateTimeOffset at)
    {
        var (infix, extension) = Conventions[kind];
        return $"{ModelName.Normalize(modelName)}-{infix}-{at.ToLocalTime():yyyyMMdd-HHmmss}{extension}";
    }

    /// <summary>
    /// The search patterns that find every prediction file belonging to <paramref name="modelName"/>,
    /// one per kind.
    /// </summary>
    /// <remarks>
    /// One pattern per kind rather than a single <c>{model}-*</c>, because model names may contain
    /// hyphens: a project with models <c>my</c> and <c>my-other</c> would have the first one's glob
    /// swallow the second one's output.
    /// </remarks>
    public static IReadOnlyList<string> PredictionSearchPatterns(string modelName)
    {
        var normalized = ModelName.Normalize(modelName);
        return [.. Conventions.Values.Select(c => $"{normalized}-{c.Infix}-*{c.Extension}")];
    }

    /// <summary>
    /// The same, for any model — for a caller asking "has this project predicted at all".
    /// </summary>
    /// <remarks>
    /// The site this exists for was looking for <c>*.csv</c>, which is the same omission as the
    /// per-model one in a different disguise: it counted a forecast but not a set of detections,
    /// and would count any stray csv a user dropped in the directory.
    /// </remarks>
    public static IReadOnlyList<string> PredictionSearchPatterns()
        => [.. Conventions.Values.Select(c => $"*-{c.Infix}-*{c.Extension}")];
}
