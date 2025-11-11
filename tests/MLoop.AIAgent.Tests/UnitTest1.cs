using MLoop.AIAgent.Infrastructure;
using Microsoft.Extensions.Logging.Abstractions;

namespace MLoop.AIAgent.Tests;

public class ConversationManagerTests
{
    [Fact]
    public void AddUserMessage_ShouldAddToHistory()
    {
        // Arrange
        var manager = new ConversationManager(NullLogger<ConversationManager>.Instance);
        var message = "Test message";

        // Act
        manager.AddUserMessage(message);

        // Assert
        var history = manager.GetHistory();
        Assert.Single(history);
        Assert.Equal("user", history[0].Speaker);
        Assert.Equal(message, history[0].Message);
    }

    [Fact]
    public void AddAgentResponse_ShouldAddToHistory()
    {
        // Arrange
        var manager = new ConversationManager(NullLogger<ConversationManager>.Instance);
        var agentName = "data-analyst";
        var response = "Test response";

        // Act
        manager.AddAgentResponse(agentName, response);

        // Assert
        var history = manager.GetHistory();
        Assert.Single(history);
        Assert.Equal(agentName, history[0].Speaker);
        Assert.Equal(response, history[0].Message);
    }

    [Fact]
    public void ClearHistory_ShouldRemoveAllMessages()
    {
        // Arrange
        var manager = new ConversationManager(NullLogger<ConversationManager>.Instance);
        manager.AddUserMessage("Message 1");
        manager.AddAgentResponse("agent", "Response 1");

        // Act
        manager.ClearHistory();

        // Assert
        Assert.Empty(manager.GetHistory());
    }

    [Fact]
    public void SetProjectContext_ShouldUpdateProjectPath()
    {
        // Arrange
        var manager = new ConversationManager(NullLogger<ConversationManager>.Instance);
        var projectPath = "/path/to/project";

        // Act
        manager.SetProjectContext(projectPath);

        // Assert
        Assert.Equal(projectPath, manager.CurrentProjectPath);
    }

    [Fact]
    public void SetDataFileContext_ShouldUpdateDataFile()
    {
        // Arrange
        var manager = new ConversationManager(NullLogger<ConversationManager>.Instance);
        var dataFile = "data.csv";

        // Act
        manager.SetDataFileContext(dataFile);

        // Assert
        Assert.Equal(dataFile, manager.CurrentDataFile);
    }
}
