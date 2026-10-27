using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// A file read to be scored by a trained model is prepared by one sequence (<c>InferenceDataPreprocessor.Prepare</c> —
/// encoding, flattening, index columns, and the columns training left out) and then laid out by one reconciliation
/// (<c>CsvDataLoader.ReconcileInferredSchemaForInference</c>). The two are a pair: a path that reconciles a layout
/// over a file nobody prepared lays out columns training removed, and the vector it builds is a slot wider than the
/// model's. Every reader after training — predict, evaluate, threshold, calibrate, export — has been wrong this way
/// when it did one and not the other.
/// </summary>
public class InferenceReadingPathContractTests
{
    private static readonly Regex Reconcile = new(@"ReconcileInferredSchemaForInference\(", RegexOptions.Compiled);
    private static readonly Regex Prepare = new(@"InferenceDataPreprocessor\.Prepare\(", RegexOptions.Compiled);

    private static bool IsDocumentation(string line) => line.TrimStart().StartsWith("//", StringComparison.Ordinal)
                                                       || line.TrimStart().StartsWith("///", StringComparison.Ordinal);

    // The file that defines the reconciliation is not a reader.
    private static IEnumerable<string> Offenders(IEnumerable<(string File, string[] Lines)> sources) =>
        from source in sources
        where Path.GetFileName(source.File) != "CsvDataLoader.cs"
        let code = source.Lines.Where(l => !IsDocumentation(l)).ToList()
        where code.Any(Reconcile.IsMatch) && !code.Any(Prepare.IsMatch)
        select source.File;

    [Fact]
    public void EveryReaderThatReconcilesALayoutPreparesTheFileFirst()
    {
        var sources = RepoSourceTree.ProductionSourceFiles("src", "tools")
            .Select(f => (File: RepoSourceTree.RelativeToRoot(f), Lines: File.ReadAllLines(f))).ToList();

        // It read the tree: both readers it exists for are among what it scanned, and each does both things.
        var readers = sources.Where(s => s.Lines.Any(l => !IsDocumentation(l) && Reconcile.IsMatch(l)) && Path.GetFileName(s.File) != "CsvDataLoader.cs")
            .Select(s => Path.GetFileName(s.File)).Order().ToList();
        Assert.Contains("EvaluationEngine.cs", readers);
        Assert.Contains("PredictionEngine.cs", readers);

        var offenders = Offenders(sources).ToList();
        Assert.True(offenders.Count == 0,
            "A file that reconciles an inference layout must prepare the data with InferenceDataPreprocessor.Prepare first:" +
            Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }

    /// <summary>The scan can see: it finds a reader that reconciles without preparing, and lets a pair, the definition and a comment pass.</summary>
    [Fact]
    public void TheScanSeesAReaderThatSkipsThePreparation()
    {
        var forbidden = ("tools/X.cs", new[] { "        CsvDataLoader.ReconcileInferredSchemaForInference(options, schema, label, path, null);" });
        var pair = ("tools/Y.cs", new[]
        {
            "        var path = InferenceDataPreprocessor.Prepare(file, label, schema, out var temps);",
            "        CsvDataLoader.ReconcileInferredSchemaForInference(options, schema, label, path, null);",
        });
        var definition = ("src/MLoop.Core/Data/CsvDataLoader.cs", new[] { "    public static void ReconcileInferredSchemaForInference(" });
        var comment = ("tools/Z.cs", new[] { "        // see ReconcileInferredSchemaForInference(options, ...) for the layout" });

        Assert.Equal(["tools/X.cs"], Offenders([forbidden, pair, definition, comment]));
    }
}
