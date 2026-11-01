using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// Every TorchSharp trainer is fitted through <c>TorchFit.Run</c>, which seeds libtorch first. Without it the first fit of a process
/// drew from a generator nobody seeded, so the same images and settings gave a different detector in every process (mAP 0, 0.19,
/// 0.31, 0.33 on one set) — a "train again" that was luck. Nothing in the type system stops the next trainer from being added with
/// a bare <c>.Fit</c>, so this reads the handlers.
/// </summary>
public class TorchFitContractTests
{
    private static readonly Regex TorchTrainer = new(
        @"Trainers\.(TextClassification|SentenceSimilarity|NamedEntityRecognition|ObjectDetection|QuestionAnswer)\(",
        RegexOptions.CultureInvariant | RegexOptions.Compiled);

    private static readonly Regex SeededFit = new(@"\bTorchFit\.Run\(", RegexOptions.CultureInvariant | RegexOptions.Compiled);
    private static readonly Regex BareGuard = new(@"\bPretrainedWeights\.Guard\(", RegexOptions.CultureInvariant | RegexOptions.Compiled);

    private static string CodeOf(string source) =>
        string.Join('\n', source.Split('\n').Where(line =>
        {
            var t = line.TrimStart();
            return !t.StartsWith("//", StringComparison.Ordinal) && !t.StartsWith("*", StringComparison.Ordinal) && !t.StartsWith("/*", StringComparison.Ordinal);
        }));

    /// <summary>What is wrong with a handlers source: a trainer count that is not the seeded-fit count, or a bare guard.</summary>
    internal static IReadOnlyList<string> Problems(string source)
    {
        var code = CodeOf(source);
        var trainers = TorchTrainer.Matches(code).Count;
        var fits = SeededFit.Matches(code).Count;
        var problems = new List<string>();
        if (trainers != fits)
            problems.Add($"{trainers} TorchSharp trainer(s) but {fits} fit(s) through TorchFit.Run");
        if (BareGuard.IsMatch(code))
            problems.Add("a fit goes through PretrainedWeights.Guard directly, unseeded");
        return problems;
    }

    [Fact]
    public void EveryTorchSharpTrainerIsFittedThroughTheSeededWrapper()
    {
        var handlers = RepoSourceTree.ProductionSourceFiles(Path.Combine("src", "MLoop.Core.DeepLearning", "Handlers"))
            .Where(f => Path.GetFileName(f) == "DeepLearningHandlers.cs")
            .ToList();
        Assert.Single(handlers);

        var source = File.ReadAllText(handlers[0]);
        Assert.True(TorchTrainer.Matches(CodeOf(source)).Count >= 5, "the scan must see the trainers it polices");
        Assert.Empty(Problems(source));
    }

    [Fact]
    public void NoOtherFileInTheModuleFitsUnderTheBareGuard()
    {
        var offenders = RepoSourceTree.ProductionSourceFiles(Path.Combine("src", "MLoop.Core.DeepLearning"))
            .Where(f => Path.GetFileName(f) is not ("TorchFit.cs" or "PretrainedWeights.cs"))
            .Where(f => BareGuard.IsMatch(CodeOf(File.ReadAllText(f))))
            .Select(RepoSourceTree.RelativeToRoot)
            .ToList();

        Assert.Empty(offenders);
    }

    [Fact]
    public void TheMatcherSeesTheShapesItForbidsAndAcceptsTheOneItAllows()
    {
        const string bare =
            "var p = mlContext.MulticlassClassification.Trainers.ObjectDetection(x);\n" +
            "model = PretrainedWeights.Guard(() => p.Fit(train));\n";
        const string seeded =
            "var p = mlContext.MulticlassClassification.Trainers.ObjectDetection(x);\n" +
            "model = TorchFit.Run(() => p.Fit(train));\n";
        const string missing =
            "var p = mlContext.MulticlassClassification.Trainers.QuestionAnswer(x);\n" +
            "var model = p.Fit(train);\n";
        const string commented =
            "// PretrainedWeights.Guard( was here\n" +
            "var p = mlContext.MulticlassClassification.Trainers.ObjectDetection(x);\n" +
            "model = TorchFit.Run(() => p.Fit(train));\n";

        Assert.NotEmpty(Problems(bare));
        Assert.NotEmpty(Problems(missing));
        Assert.Empty(Problems(seeded));
        Assert.Empty(Problems(commented));
    }
}
