using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Training;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// <see cref="EnsembleChain"/> — fitted members joined in one chain that answers with the weighted mean, saves as one model
/// file and reads back to the same answers, and whose members see what they were fitted on.
/// </summary>
public sealed class EnsembleChainTests
{
    private sealed class BinaryRow
    {
        public float A { get; set; }
        public float B { get; set; }
        public bool Y { get; set; }
    }

    private sealed class RegressionRow
    {
        public float A { get; set; }
        public float B { get; set; }
        public float Y { get; set; }
    }

    private static List<BinaryRow> BinaryRows(int count, int seed, float scale = 1)
    {
        var random = new Random(seed);
        return [.. Enumerable.Range(0, count).Select(_ =>
        {
            var a = (float)random.NextDouble() * 10;
            var b = (float)random.NextDouble() * 5;
            return new BinaryRow { A = a * scale, B = b, Y = 3 * a + 2 * b + random.NextDouble() * 4 > 20 };
        })];
    }

    private static List<RegressionRow> RegressionRows(int count, int seed)
    {
        var random = new Random(seed);
        return [.. Enumerable.Range(0, count).Select(_ =>
        {
            var a = (float)random.NextDouble() * 10;
            var b = (float)random.NextDouble() * 5;
            return new RegressionRow { A = a, B = b, Y = 3 * a + 2 * b + (float)random.NextDouble() * 4 };
        })];
    }

    private static Func<MLContext, IEstimator<ITransformer>> Logistic(Func<MLContext, IEstimator<ITransformer>>? prefix = null)
        => ml => (prefix?.Invoke(ml) ?? new EstimatorChain<ITransformer>())
            .Append(ml.Transforms.Concatenate("Features", "A", "B"))
            .Append(ml.BinaryClassification.Trainers.SingleThreadSdcaLogisticRegression(labelColumnName: "Y"));

    private static Func<MLContext, IEstimator<ITransformer>> Lbfgs()
        => ml => ml.Transforms.Concatenate("Features", "A", "B")
            .Append(ml.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: "Y"));

    // The chain up to its members' answers (everything ahead of the blend's first lambda): the blend's arithmetic is checked against these,
    // not against members fitted again — a context's random state moves with every Fit, so a second fit is a different model.
    private static TransformerChain<ITransformer> MembersOf(TransformerChain<ITransformer> chain, int count)
        => new(chain.TakeWhile(step => !step.GetType().Name.Contains("Expression", StringComparison.Ordinal)).ToArray());

    private static float[] Column(IDataView view, string name) => [.. view.GetColumn<float>(name)];

    [Fact]
    public void Binary_AnswersTheWeightedMeanOfItsMembersProbabilities()
    {
        var ml = new MLContext(seed: 1);
        var train = ml.Data.LoadFromEnumerable(BinaryRows(600, 1));
        var test = ml.Data.LoadFromEnumerable(BinaryRows(200, 2));

        var chain = EnsembleChain.Fit(ml, [new EnsembleMember(Logistic(), 1), new EnsembleMember(Lbfgs(), 2)], train, EnsembleKind.Binary);

        var answers = MembersOf(chain, count: 2).Transform(test);
        var first = Column(answers, "Member0");
        var second = Column(answers, "Member1");
        var blended = chain.Transform(test);
        var probability = Column(blended, "Probability");
        for (var i = 0; i < probability.Length; i++)
            Assert.Equal((first[i] + 2 * second[i]) / 3, probability[i], 4);
        Assert.Equal(probability.Select(p => p >= 0.5f), blended.GetColumn<bool>("PredictedLabel"));
        // The score is the log-odds of the probability: zero exactly where the probability is one half.
        var score = Column(blended, "Score");
        for (var i = 0; i < probability.Length; i++)
        {
            var p = Math.Clamp(probability[i], 1e-6f, 1 - 1e-6f);
            Assert.Equal(Math.Log(p / (1 - p)), score[i], 3);
            Assert.Equal(probability[i] >= 0.5f, score[i] >= 0);
        }
    }

    [Fact]
    public void Binary_SavesAsOneModelFile_AndReadsBackToTheSameAnswersInAFreshContext()
    {
        var ml = new MLContext(seed: 1);
        var train = ml.Data.LoadFromEnumerable(BinaryRows(600, 1));
        var test = ml.Data.LoadFromEnumerable(BinaryRows(100, 2));
        var chain = EnsembleChain.Fit(ml, [new EnsembleMember(Logistic(), 1), new EnsembleMember(Lbfgs(), 1)], train, EnsembleKind.Binary);
        var path = Path.Combine(Path.GetTempPath(), $"ensemble-{Guid.NewGuid():N}.zip");
        try
        {
            ml.Model.Save(chain, train.Schema, path);
            var fresh = new MLContext();
            var loaded = fresh.Model.Load(path, out _);

            var before = chain.Transform(test);
            var after = loaded.Transform(fresh.Data.LoadFromEnumerable(BinaryRows(100, 2)));
            Assert.Equal(Column(before, "Probability"), Column(after, "Probability"));
            Assert.Equal(before.GetColumn<bool>("PredictedLabel"), after.GetColumn<bool>("PredictedLabel"));
            Assert.DoesNotContain(after.Schema, c => !c.IsHidden && c.Name.StartsWith("Member", StringComparison.Ordinal));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Regression_AnswersTheWeightedMeanOfItsMembersValues()
    {
        var ml = new MLContext(seed: 1);
        var train = ml.Data.LoadFromEnumerable(RegressionRows(600, 3));
        var test = ml.Data.LoadFromEnumerable(RegressionRows(100, 4));
        Func<MLContext, IEstimator<ITransformer>> ols = m => m.Transforms.Concatenate("Features", "A", "B")
            .Append(m.Regression.Trainers.SingleThreadSdca(labelColumnName: "Y"));
        Func<MLContext, IEstimator<ITransformer>> lbfgs = m => m.Transforms.Concatenate("Features", "A", "B")
            .Append(m.Regression.Trainers.LbfgsPoissonRegression(labelColumnName: "Y"));

        var chain = EnsembleChain.Fit(ml, [new EnsembleMember(ols, 3), new EnsembleMember(lbfgs, 1)], train, EnsembleKind.Regression);

        var answers = MembersOf(chain, count: 2).Transform(test);
        var first = Column(answers, "Member0");
        var second = Column(answers, "Member1");
        var score = Column(chain.Transform(test), "Score");
        for (var i = 0; i < score.Length; i++)
            Assert.Equal((3 * first[i] + second[i]) / 4, score[i], 3);
    }

    [Fact]
    public void AMemberThatRewritesAnInputInPlace_ReadsWhatItWasFittedOn()
    {
        // Both members normalize A in place ahead of their trainer. The second sits behind the first, so it reads A already
        // normalized; fitted on the raw column its normalization would be applied twice at scoring and not at training.
        // Fitted behind the first it is a normalization of a normalized column — the identity — so a blend of the same
        // member twice answers what that member answers alone.
        var ml = new MLContext(seed: 1);
        var train = ml.Data.LoadFromEnumerable(BinaryRows(600, 1, scale: 40));
        var test = ml.Data.LoadFromEnumerable(BinaryRows(200, 2, scale: 40));
        var normalized = Logistic(m => m.Transforms.NormalizeMeanVariance("A"));

        var chain = EnsembleChain.Fit(ml, [new EnsembleMember(normalized, 1), new EnsembleMember(normalized, 1)], train, EnsembleKind.Binary);

        var alone = Column(normalized(new MLContext(seed: 1)).Fit(train).Transform(test), "Probability");
        var blended = Column(chain.Transform(test), "Probability");
        // Not equal to the digit — the second member's shuffling reads a context the first has used — but close; a second
        // normalization fitted on the raw column moves the answers by far more than this.
        var worst = alone.Select((a, i) => Math.Abs(a - blended[i])).Max();
        Assert.True(worst < 0.03, $"The blend of a member with itself differs from the member by {worst:0.####}.");
    }

    [Fact]
    public void ABlend_IsRefusedForOnnx_InWordsThatNameTheEnsembleAndTheWayOut()
    {
        var ml = new MLContext(seed: 1);
        var train = ml.Data.LoadFromEnumerable(BinaryRows(300, 1));
        var chain = EnsembleChain.Fit(ml, [new EnsembleMember(Logistic(), 1), new EnsembleMember(Lbfgs(), 1)], train, EnsembleKind.Binary);

        Assert.Contains("ensemble", MLoop.Core.Export.OnnxExport.FindNotExportable(chain), StringComparison.Ordinal);
        using var sink = new MemoryStream();
        var ex = Assert.Throws<MLoop.Core.Export.OnnxExportException>(() =>
            MLoop.Core.Export.OnnxExport.Write(ml, chain, train, new TextLoader.Options(), ["A", "B", "Y"], "Y", sink));
        Assert.Contains("ensemble", ex.Message, StringComparison.Ordinal);
        Assert.Contains("single best model", ex.Message, StringComparison.Ordinal);
        Assert.Equal(0, sink.Length);
    }

    [Fact]
    public void ASingleModel_IsNotRefusedAsABlend()
    {
        var ml = new MLContext(seed: 1);
        var train = ml.Data.LoadFromEnumerable(BinaryRows(300, 1));

        Assert.Null(MLoop.Core.Export.OnnxExport.FindNotExportable(Logistic()(ml).Fit(train)));
    }

    private sealed class CategoricalRow
    {
        public string Kind { get; set; } = "";
        public float A { get; set; }
        public bool Y { get; set; }
    }

    [Fact]
    public void MembersThatEncodeAColumnInPlace_EachReadTheRawColumn()
    {
        // Featurizing a string column under its own name turns it into a vector. A member behind such a member must still find
        // the string — every member reads the raw columns, whatever an earlier one made of them in place.
        var ml = new MLContext(seed: 1);
        var random = new Random(5);
        var kinds = new[] { "red", "green", "blue" };
        var rows = Enumerable.Range(0, 400).Select(_ =>
        {
            var kind = kinds[random.Next(3)];
            var a = (float)random.NextDouble();
            return new CategoricalRow { Kind = kind, A = a, Y = (kind == "red") ^ (a > 0.8f) };
        }).ToList();
        var data = ml.Data.LoadFromEnumerable(rows);
        Func<MLContext, IEstimator<ITransformer>> encoded = m => m.Transforms.Text.FeaturizeText("Kind")
            .Append(m.Transforms.Concatenate("Features", "Kind", "A"))
            .Append(m.BinaryClassification.Trainers.SingleThreadSdcaLogisticRegression(labelColumnName: "Y"));

        var chain = EnsembleChain.Fit(ml, [new EnsembleMember(encoded, 1), new EnsembleMember(encoded, 1)], data, EnsembleKind.Binary);

        var probability = Column(chain.Transform(ml.Data.LoadFromEnumerable(rows.Take(50))), "Probability");
        Assert.Equal(50, probability.Length);
        Assert.All(probability, p => Assert.InRange(p, 0f, 1f));
        // Scoring rows that arrive as the raw table does — the string column — works the same on a fresh context.
        var path = Path.Combine(Path.GetTempPath(), $"ensemble-cat-{Guid.NewGuid():N}.zip");
        try
        {
            ml.Model.Save(chain, data.Schema, path);
            var fresh = new MLContext();
            var again = Column(fresh.Model.Load(path, out _).Transform(fresh.Data.LoadFromEnumerable(rows.Take(50))), "Probability");
            Assert.Equal(probability, again);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ABlendWhoseTimeIsUp_StopsBetweenMembers()
    {
        var ml = new MLContext(seed: 1);
        var train = ml.Data.LoadFromEnumerable(BinaryRows(200, 1));
        var asked = 0;

        var ex = Assert.Throws<EnsembleOverBudgetException>(() => EnsembleChain.Fit(
            ml, [new EnsembleMember(Logistic(), 1), new EnsembleMember(Lbfgs(), 1), new EnsembleMember(Logistic(), 1)], train, EnsembleKind.Binary,
            giveUp: () => ++asked >= 2));

        Assert.Equal(2, ex.Fitted);
        Assert.Contains("2 of the 3 members", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WeightedMean_PrintsIntegerCountsOnly()
        => Assert.Equal("(x0, x1, x2) => (2 * x0 + 1 * x1 + 3 * x2) / 6", EnsembleChain.WeightedMean([2, 1, 3]));

    [Fact]
    public void ABlendWithNoMemberThatCounts_OrMoreThanTheMostItHolds_IsRefused()
    {
        var ml = new MLContext(seed: 1);
        var train = ml.Data.LoadFromEnumerable(BinaryRows(100, 1));
        Assert.Throws<ArgumentException>(() => EnsembleChain.Fit(ml, [new EnsembleMember(Logistic(), 0)], train, EnsembleKind.Binary));
        var tooMany = Enumerable.Range(0, EnsembleChain.MaxMembers + 1).Select(_ => new EnsembleMember(Logistic(), 1)).ToArray();
        Assert.Throws<ArgumentException>(() => EnsembleChain.Fit(ml, tooMany, train, EnsembleKind.Binary));
    }
}
