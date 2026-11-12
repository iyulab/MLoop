using FluentAssertions;
using MLoop.Core.Storage;

namespace MLoop.API.Tests;

/// <summary>
/// Which model a training job trains into. An image built for one model (<c>MLOOP_MODEL_NAME</c>) answered every other request
/// from that model, but <c>POST /train</c> without a name trained <c>default</c> - the one endpoint that built the name itself.
/// </summary>
public class TrainingJobModelNameTests : IDisposable
{
    private readonly string? _saved = Environment.GetEnvironmentVariable(ModelName.EnvironmentVariable);

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(ModelName.EnvironmentVariable, _saved);
        GC.SuppressFinalize(this);
    }

    private static TrainingJobRequest Request(string? name) => new("datasets/t.csv", "y", "binary-classification", name);

    [Fact]
    public void ARequestThatNamesNoModel_TrainsTheOneTheEnvironmentNames()
    {
        Environment.SetEnvironmentVariable(ModelName.EnvironmentVariable, "churn");

        Request(null).ResolveModelName().Should().Be("churn");
        Request("  ").ResolveModelName().Should().Be("churn");
    }

    [Fact]
    public void ARequestThatNamesAModel_TrainsThat_AndWithoutTheVariableTheDefault()
    {
        Environment.SetEnvironmentVariable(ModelName.EnvironmentVariable, "churn");
        Request(" Fraud ").ResolveModelName().Should().Be("fraud");

        Environment.SetEnvironmentVariable(ModelName.EnvironmentVariable, null);
        Request(null).ResolveModelName().Should().Be("default");
    }

    [Fact]
    public void ANameThatIsNotADirectoryName_IsRefused()
    {
        Environment.SetEnvironmentVariable(ModelName.EnvironmentVariable, null);

        var act = () => Request("../../escape").ResolveModelName();

        act.Should().Throw<InvalidModelNameException>();
    }
}
