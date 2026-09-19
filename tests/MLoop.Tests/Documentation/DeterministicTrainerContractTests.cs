using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// A test fixture that trains an SDCA model does it on one thread, through
/// <c>DeterministicTrainers</c>.
/// </summary>
/// <remarks>
/// ML.NET's SDCA defaults to several threads, and on the tiny fixtures these suites train it does
/// not converge — it runs to its iteration limit, and its model differs between runs of the same
/// seed. On a starved machine that turned single fits into many minutes, and three suites stopped
/// at them. The helper pins one thread; this scan keeps a new fixture from bringing the default back.
/// </remarks>
public class DeterministicTrainerContractTests
{
    /// <summary>A call to one of ML.NET's SDCA trainer factories on a trainer catalog.</summary>
    private static readonly Regex BareSdcaCall =
        new(@"\.(?:Sdca|SdcaLogisticRegression|SdcaMaximumEntropy|SdcaNonCalibrated)\s*\(", RegexOptions.Compiled);

    private static bool CallsBareSdca(string source) => BareSdcaCall.IsMatch(source);

    [Fact]
    public void NoTestFixtureTrainsSdcaOnTheDefaultThreadCount()
    {
        var scanned = RepoSourceTree.SourceFiles(nameof(DeterministicTrainerContractTests))
            .Where(f => Path.GetFileName(f) != "DeterministicTrainers.cs")
            .ToList();

        // The scan reaches the fixtures it is about: these files train SDCA models through the
        // helper, so an empty or misrouted scan cannot pass for compliance.
        Assert.Contains(scanned, f => Path.GetFileName(f) == "PredictConfidenceColumnTests.cs");
        Assert.Contains(scanned, f => Path.GetFileName(f) == "SchemaVocabularyDiagnosisTests.cs");
        Assert.Contains(scanned, f => Path.GetFileName(f) == "ModelCacheTests.cs");

        var offenders = scanned
            .Where(f => CallsBareSdca(File.ReadAllText(f)))
            .Select(RepoSourceTree.Relative)
            .Order(StringComparer.Ordinal)
            .ToList();

        Assert.True(offenders.Count == 0,
            "SDCA fixtures train on one thread (SingleThreadSdca, SingleThreadSdcaLogisticRegression, "
            + "SingleThreadSdcaMaximumEntropy):\n" + string.Join('\n', offenders));
    }

    [Fact]
    public void TheScanSeesTheShapeItForbidsAndAcceptsTheOneItWants()
    {
        Assert.True(CallsBareSdca("ml.Regression.Trainers.Sdca(labelColumnName: \"Y\")"));
        Assert.True(CallsBareSdca("ml.BinaryClassification.Trainers.SdcaLogisticRegression(\n    labelColumnName: \"L\")"));
        Assert.True(CallsBareSdca("ml.MulticlassClassification.Trainers.SdcaMaximumEntropy (\"Species\")"));

        Assert.False(CallsBareSdca("ml.Regression.Trainers.SingleThreadSdca(labelColumnName: \"Y\")"));
        Assert.False(CallsBareSdca("ml.MulticlassClassification.Trainers.SingleThreadSdcaMaximumEntropy(\"Species\")"));
    }
}
