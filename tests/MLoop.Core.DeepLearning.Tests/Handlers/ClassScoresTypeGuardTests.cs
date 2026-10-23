using MLoop.Core.Models;

namespace MLoop.Core.DeepLearning.Tests.Handlers;

/// <summary>
/// <see cref="ClassScores"/> knows the logit-scoring text classifier by its type name, because MLoop.Core does not
/// reference TorchSharp. Should a package update rename the type, the name would match nothing and every text
/// classification would silently answer in logits again — so the name is held to the compiled type here.
/// </summary>
public class ClassScoresTypeGuardTests
{
    [Fact]
    public void TheLogitTransformerName_IsTheTextClassifiersType()
        => Assert.Equal(typeof(Microsoft.ML.TorchSharp.NasBert.TextClassificationTransformer).FullName, ClassScores.LogitTransformer);
}
