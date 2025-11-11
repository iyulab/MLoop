using System.CommandLine;
using MLoop.CLI.Commands;

namespace MLoop.AIAgent.Tests;

public class AgentCommandTests
{
    [Fact]
    public void AgentCommand_Create_ReturnsCommand()
    {
        // Act
        var command = AgentCommand.Create();

        // Assert
        Assert.NotNull(command);
        Assert.Equal("agent", command.Name);
        Assert.Equal("Conversational AI agent for ML project management", command.Description);
    }

    [Fact]
    public void AgentCommand_HasQueryArgument()
    {
        // Act
        var command = AgentCommand.Create();

        // Assert
        var queryArg = command.Arguments.FirstOrDefault(a => a.Name == "query");
        Assert.NotNull(queryArg);
    }

    [Fact]
    public void AgentCommand_HasAgentOption()
    {
        // Act
        var command = AgentCommand.Create();

        // Assert
        var agentOption = command.Options.FirstOrDefault(o => o.Name == "agent");
        Assert.NotNull(agentOption);
    }

    [Fact]
    public void AgentCommand_HasInteractiveOption()
    {
        // Act
        var command = AgentCommand.Create();

        // Assert
        var interactiveOption = command.Options.FirstOrDefault(o => o.Name == "interactive");
        Assert.NotNull(interactiveOption);
    }

    [Fact]
    public void AgentCommand_HasProjectPathOption()
    {
        // Act
        var command = AgentCommand.Create();

        // Assert
        var projectPathOption = command.Options.FirstOrDefault(o => o.Name == "project");
        Assert.NotNull(projectPathOption);
    }
}
