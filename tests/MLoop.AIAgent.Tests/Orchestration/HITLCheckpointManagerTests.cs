// Copyright (c) IYULab. All rights reserved.
// Licensed under the MIT License.

using Ironbees.Core.Goals;
using MLoop.AIAgent.Core.Orchestration;

namespace MLoop.AIAgent.Tests.Orchestration;

public class HITLCheckpointManagerTests
{
    [Fact]
    public void CheckpointDefinitions_ContainsAllHitlStates()
    {
        // Assert
        Assert.True(HITLCheckpointManager.CheckpointDefinitions.ContainsKey(OrchestrationState.DataAnalysisReview));
        Assert.True(HITLCheckpointManager.CheckpointDefinitions.ContainsKey(OrchestrationState.ModelSelectionReview));
        Assert.True(HITLCheckpointManager.CheckpointDefinitions.ContainsKey(OrchestrationState.PreprocessingReview));
        Assert.True(HITLCheckpointManager.CheckpointDefinitions.ContainsKey(OrchestrationState.TrainingReview));
        Assert.True(HITLCheckpointManager.CheckpointDefinitions.ContainsKey(OrchestrationState.DeploymentReview));
    }

    [Fact]
    public void CheckpointDefinitions_HaveUniqueIds()
    {
        // Act
        var ids = HITLCheckpointManager.CheckpointDefinitions.Values.Select(d => d.Id).ToList();

        // Assert
        Assert.Equal(ids.Count, ids.Distinct().Count());
    }

    [Fact]
    public void CheckpointDefinitions_HaveValidPhaseNumbers()
    {
        // Assert
        foreach (var definition in HITLCheckpointManager.CheckpointDefinitions.Values)
        {
            Assert.InRange(definition.Phase, 1, 5);
        }
    }

    [Fact]
    public void CheckpointDefinitions_HaveDefaultOptions()
    {
        // Assert
        foreach (var definition in HITLCheckpointManager.CheckpointDefinitions.Values)
        {
            Assert.NotEmpty(definition.DefaultOptions);
            Assert.Contains(definition.DefaultOptions, o => o.IsDefault);
        }
    }

    [Theory]
    [InlineData(OrchestrationState.TrainingReview)]
    [InlineData(OrchestrationState.DeploymentReview)]
    public void CheckpointDefinitions_CriticalCheckpoints_RequireExplicitApproval(OrchestrationState state)
    {
        // Arrange
        var definition = HITLCheckpointManager.CheckpointDefinitions[state];

        // Assert
        Assert.True(definition.RequiresExplicitApproval);
        Assert.Equal(double.MaxValue, definition.AutoApprovalThreshold);
    }

    [Fact]
    public void ShouldTriggerHitl_ReturnsFalse_WhenSkipHitlEnabled()
    {
        // Arrange
        var manager = new HITLCheckpointManager();
        var options = new OrchestrationOptions { SkipHitl = true };

        // Act
        var result = manager.ShouldTriggerHitl(OrchestrationState.DataAnalysisReview, 0.5, options);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void ShouldTriggerHitl_ReturnsFalse_WhenNotHitlCheckpoint()
    {
        // Arrange
        var manager = new HITLCheckpointManager();
        var options = new OrchestrationOptions();

        // Act
        var result = manager.ShouldTriggerHitl(OrchestrationState.DataAnalysis, 0.5, options);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void ShouldTriggerHitl_ReturnsTrue_WhenRequiresExplicitApproval()
    {
        // Arrange
        var manager = new HITLCheckpointManager();
        var options = new OrchestrationOptions { AutoApproveHighConfidence = true };

        // Act - TrainingReview always requires explicit approval
        var result = manager.ShouldTriggerHitl(OrchestrationState.TrainingReview, 1.0, options);

        // Assert
        Assert.True(result);
    }

    [Theory]
    [InlineData(OrchestrationState.DataAnalysisReview, 0.90, false)] // Above 0.85 threshold
    [InlineData(OrchestrationState.DataAnalysisReview, 0.80, true)]  // Below 0.85 threshold
    [InlineData(OrchestrationState.ModelSelectionReview, 0.85, false)] // At 0.80 threshold (>=)
    [InlineData(OrchestrationState.ModelSelectionReview, 0.75, true)]  // Below 0.80 threshold
    [InlineData(OrchestrationState.PreprocessingReview, 0.80, false)]  // Above 0.75 threshold
    [InlineData(OrchestrationState.PreprocessingReview, 0.70, true)]   // Below 0.75 threshold
    public void ShouldTriggerHitl_RespectsAutoApprovalThreshold(
        OrchestrationState state, double confidence, bool expectedTrigger)
    {
        // Arrange
        var manager = new HITLCheckpointManager();
        var options = new OrchestrationOptions { AutoApproveHighConfidence = true };

        // Act
        var result = manager.ShouldTriggerHitl(state, confidence, options);

        // Assert
        Assert.Equal(expectedTrigger, result);
    }

    [Fact]
    public void ShouldTriggerHitl_ReturnsTrue_WhenAutoApproveDisabled()
    {
        // Arrange
        var manager = new HITLCheckpointManager();
        var options = new OrchestrationOptions { AutoApproveHighConfidence = false };

        // Act - Even with high confidence, should trigger if auto-approve is disabled
        var result = manager.ShouldTriggerHitl(OrchestrationState.DataAnalysisReview, 0.99, options);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void ShouldTriggerHitl_UsesMinimumOfDefinitionAndOptionThreshold()
    {
        // Arrange
        var manager = new HITLCheckpointManager();
        var options = new OrchestrationOptions
        {
            AutoApproveHighConfidence = true,
            AutoApprovalThreshold = 0.70 // Lower than DataAnalysisReview's 0.85
        };

        // Act - With 0.75 confidence, should NOT trigger (above 0.70 option threshold)
        var result = manager.ShouldTriggerHitl(OrchestrationState.DataAnalysisReview, 0.75, options);

        // Assert
        Assert.False(result); // 0.75 >= min(0.85, 0.70) = 0.70
    }

    [Fact]
    public void ShouldTriggerHitl_UsesAgenticSettings_WhenProvided()
    {
        // Arrange
        var agenticSettings = new AgenticSettings
        {
            Hitl = new HitlSettings
            {
                Policy = HitlPolicy.OnUncertainty,
                UncertaintyThreshold = 0.8,
                Checkpoints = ["data-analysis-review"]
            }
        };
        var manager = new HITLCheckpointManager(agenticSettings);
        var options = new OrchestrationOptions();

        // Act
        var result = manager.ShouldTriggerHitl(OrchestrationState.DataAnalysisReview, 0.75, options);

        // Assert - Should trigger because confidence (0.75) < uncertainty threshold (0.8)
        Assert.True(result);
    }

    [Fact]
    public void CreateHitlRequest_ReturnsValidEvent()
    {
        // Arrange
        var manager = new HITLCheckpointManager();
        var context = CreateTestContext();

        // Act
        var request = manager.CreateHitlRequest(OrchestrationState.DataAnalysisReview, context);

        // Assert
        Assert.Equal(context.SessionId, request.SessionId);
        Assert.Equal("data-analysis-review", request.CheckpointId);
        Assert.Equal("Data Analysis Review", request.CheckpointName);
        Assert.NotEmpty(request.Question);
        Assert.NotEmpty(request.Options);
    }

    [Fact]
    public void CreateHitlRequest_IncludesContextData()
    {
        // Arrange
        var manager = new HITLCheckpointManager();
        var context = CreateTestContext();
        context.DataAnalysis = new DataAnalysisResult
        {
            RowCount = 1000,
            ColumnCount = 15,
            DetectedTargetColumn = "target",
            InferredTaskType = "BinaryClassification",
            DataQualityScore = 0.95,
            Columns = [],
            Issues = [],
            Recommendations = []
        };

        // Act
        var request = manager.CreateHitlRequest(OrchestrationState.DataAnalysisReview, context);

        // Assert
        Assert.Contains("row_count", request.Context.Keys);
        Assert.Contains("column_count", request.Context.Keys);
        Assert.Contains("target_column", request.Context.Keys);
    }

    [Theory]
    [InlineData("approve", HitlResponseAction.Proceed)]
    [InlineData("modify", HitlResponseAction.Modify)]
    [InlineData("reanalyze", HitlResponseAction.Retry)]
    [InlineData("cancel", HitlResponseAction.Cancel)]
    public void ProcessResponse_DataAnalysisReview_ReturnsCorrectAction(
        string optionId, HitlResponseAction expectedAction)
    {
        // Arrange
        var manager = new HITLCheckpointManager();
        var context = CreateTestContext();

        // Act
        var action = manager.ProcessResponse(OrchestrationState.DataAnalysisReview, optionId, context);

        // Assert
        Assert.Equal(expectedAction, action);
    }

    [Theory]
    [InlineData("approve", HitlResponseAction.Proceed)]
    [InlineData("modify", HitlResponseAction.Modify)]
    [InlineData("skip-training", HitlResponseAction.Skip)]
    [InlineData("cancel", HitlResponseAction.Cancel)]
    public void ProcessResponse_ModelSelectionReview_ReturnsCorrectAction(
        string optionId, HitlResponseAction expectedAction)
    {
        // Arrange
        var manager = new HITLCheckpointManager();
        var context = CreateTestContext();

        // Act
        var action = manager.ProcessResponse(OrchestrationState.ModelSelectionReview, optionId, context);

        // Assert
        Assert.Equal(expectedAction, action);
    }

    [Theory]
    [InlineData("approve", HitlResponseAction.Proceed)]
    [InlineData("modify", HitlResponseAction.Modify)]
    [InlineData("skip", HitlResponseAction.Skip)]
    [InlineData("cancel", HitlResponseAction.Cancel)]
    public void ProcessResponse_PreprocessingReview_ReturnsCorrectAction(
        string optionId, HitlResponseAction expectedAction)
    {
        // Arrange
        var manager = new HITLCheckpointManager();
        var context = CreateTestContext();

        // Act
        var action = manager.ProcessResponse(OrchestrationState.PreprocessingReview, optionId, context);

        // Assert
        Assert.Equal(expectedAction, action);
    }

    [Theory]
    [InlineData("approve", HitlResponseAction.Proceed)]
    [InlineData("retrain", HitlResponseAction.Retry)]
    [InlineData("select-other", HitlResponseAction.Modify)]
    [InlineData("cancel", HitlResponseAction.Cancel)]
    public void ProcessResponse_TrainingReview_ReturnsCorrectAction(
        string optionId, HitlResponseAction expectedAction)
    {
        // Arrange
        var manager = new HITLCheckpointManager();
        var context = CreateTestContext();

        // Act
        var action = manager.ProcessResponse(OrchestrationState.TrainingReview, optionId, context);

        // Assert
        Assert.Equal(expectedAction, action);
    }

    [Theory]
    [InlineData("deploy", HitlResponseAction.Deploy)]
    [InlineData("export", HitlResponseAction.Export)]
    [InlineData("save", HitlResponseAction.Save)]
    [InlineData("cancel", HitlResponseAction.Cancel)]
    public void ProcessResponse_DeploymentReview_ReturnsCorrectAction(
        string optionId, HitlResponseAction expectedAction)
    {
        // Arrange
        var manager = new HITLCheckpointManager();
        var context = CreateTestContext();

        // Act
        var action = manager.ProcessResponse(OrchestrationState.DeploymentReview, optionId, context);

        // Assert
        Assert.Equal(expectedAction, action);
    }

    [Fact]
    public void ProcessResponse_Cancel_AlwaysReturnsCancel()
    {
        // Arrange
        var manager = new HITLCheckpointManager();
        var context = CreateTestContext();

        // Act & Assert - cancel works for all states
        foreach (var state in HITLCheckpointManager.CheckpointDefinitions.Keys)
        {
            var action = manager.ProcessResponse(state, "cancel", context);
            Assert.Equal(HitlResponseAction.Cancel, action);
        }
    }

    [Fact]
    public void GetTimeoutActionDescription_ReturnsDescription()
    {
        // Arrange
        var agenticSettings = new AgenticSettings
        {
            Hitl = new HitlSettings
            {
                TimeoutAction = HitlTimeoutAction.Pause
            }
        };
        var manager = new HITLCheckpointManager(agenticSettings);

        // Act
        var description = manager.GetTimeoutActionDescription();

        // Assert
        Assert.NotEmpty(description);
    }

    // Helper methods
    private static OrchestrationContext CreateTestContext()
    {
        return new OrchestrationContext
        {
            SessionId = $"test-{Guid.NewGuid():N}",
            DataFilePath = "/test/data.csv",
            Options = new OrchestrationOptions(),
            StartedAt = DateTimeOffset.UtcNow
        };
    }
}
