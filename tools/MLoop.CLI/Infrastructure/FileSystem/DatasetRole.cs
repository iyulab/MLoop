namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>The part a data file plays in a project: what a command reads when the user names no file.</summary>
public enum DatasetRole
{
    /// <summary>The data a model is trained on (<c>data.train</c>, else <c>datasets/train.csv</c>).</summary>
    Train,

    /// <summary>The labeled rows a model is scored on (<c>data.test</c>, else <c>datasets/test.csv</c>).</summary>
    Test,

    /// <summary>The rows a model answers (<c>data.predict</c>, else <c>datasets/predict.csv</c>).</summary>
    Predict,
}
