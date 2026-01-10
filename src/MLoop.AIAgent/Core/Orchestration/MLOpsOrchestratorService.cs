// Copyright (c) IYULab. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.CompilerServices;
using Ironbees.Core.Goals;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using MLoop.AIAgent.Core.Models;

namespace MLoop.AIAgent.Core.Orchestration;

/// <summary>
/// Main orchestrator service for end-to-end ML lifecycle management.
/// Single entry point: provide a data file → get a deployed model.
///
/// Workflow:
/// START → DATA_ANALYSIS → [HITL1] → MODEL_RECOMMENDATION → [HITL2]
///   → PREPROCESSING → [HITL3] → TRAINING → [HITL4] → EVALUATION
///   → [HITL5] → DEPLOYMENT → COMPLETED
/// </summary>
public class MLOpsOrchestratorService
{
    private readonly IChatClient _chatClient;
    private readonly OrchestrationSessionStore _sessionStore;
    private readonly AgentCoordinator _agentCoordinator;
    private readonly HITLCheckpointManager _hitlManager;
    private readonly MLoopProjectManager _projectManager;
    private readonly ILogger<MLOpsOrchestratorService>? _logger;

    /// <summary>
    /// Delegate for handling HITL interactions.
    /// </summary>
    /// <param name="request">HITL request event with question and options.</param>
    /// <returns>User's selected option ID and optional comment.</returns>
    public delegate Task<(string OptionId, string? Comment)> HitlHandler(HitlRequestedEvent request);

    public MLOpsOrchestratorService(
        IChatClient chatClient,
        OrchestrationSessionStore? sessionStore = null,
        AgenticSettings? agenticSettings = null,
        MLoopProjectManager? projectManager = null,
        ILogger<MLOpsOrchestratorService>? logger = null)
    {
        _chatClient = chatClient;
        _sessionStore = sessionStore ?? new OrchestrationSessionStore();
        _agentCoordinator = new AgentCoordinator(chatClient, logger as ILogger<AgentCoordinator>);
        _hitlManager = new HITLCheckpointManager(agenticSettings);
        _projectManager = projectManager ?? new MLoopProjectManager();
        _logger = logger;
    }

    /// <summary>
    /// Executes the full MLOps orchestration workflow.
    /// </summary>
    /// <param name="dataFilePath">Path to the input data file.</param>
    /// <param name="options">Orchestration options.</param>
    /// <param name="hitlHandler">Handler for HITL interactions. If null, pauses at checkpoints.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Stream of orchestration events.</returns>
    public async IAsyncEnumerable<OrchestrationEvent> ExecuteAsync(
        string dataFilePath,
        OrchestrationOptions? options = null,
        HitlHandler? hitlHandler = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= new OrchestrationOptions();

        // Validate input
        if (!File.Exists(dataFilePath))
        {
            yield return new OrchestrationFailedEvent
            {
                SessionId = "N/A",
                Error = $"Data file not found: {dataFilePath}",
                CanResume = false
            };
            yield break;
        }

        // Create context and session
        var context = new OrchestrationContext
        {
            DataFilePath = Path.GetFullPath(dataFilePath),
            Options = options,
            AgenticSettings = options.AgenticSettings
        };

        var session = new OrchestrationSession
        {
            SessionId = context.SessionId,
            Context = context
        };

        _logger?.LogInformation("Starting orchestration. Session: {SessionId}, Data: {DataFile}",
            context.SessionId, dataFilePath);

        // Save session first so it's accessible immediately
        await _sessionStore.SaveSessionAsync(session, cancellationToken);

        // Emit started event
        yield return new OrchestrationStartedEvent
        {
            SessionId = context.SessionId,
            DataFilePath = context.DataFilePath,
            Options = options
        };

        // Execute workflow
        await foreach (var evt in ExecuteWorkflowAsync(session, hitlHandler, cancellationToken))
        {
            yield return evt;
        }
    }

    /// <summary>
    /// Resumes a paused or failed orchestration session.
    /// </summary>
    public async IAsyncEnumerable<OrchestrationEvent> ResumeAsync(
        string sessionId,
        HitlHandler? hitlHandler = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var session = await _sessionStore.LoadSessionAsync(sessionId, cancellationToken);

        if (session == null)
        {
            yield return new OrchestrationFailedEvent
            {
                SessionId = sessionId,
                Error = $"Session not found: {sessionId}",
                CanResume = false
            };
            yield break;
        }

        if (!session.CanResume())
        {
            yield return new OrchestrationFailedEvent
            {
                SessionId = sessionId,
                Error = "Session cannot be resumed",
                FailedAtState = session.Context.CurrentState,
                CanResume = false
            };
            yield break;
        }

        _logger?.LogInformation("Resuming orchestration. Session: {SessionId}, State: {State}",
            sessionId, session.Context.CurrentState);

        session.Resume();
        await _sessionStore.SaveSessionAsync(session, cancellationToken);

        await foreach (var evt in ExecuteWorkflowAsync(session, hitlHandler, cancellationToken))
        {
            yield return evt;
        }
    }

    /// <summary>
    /// Lists all orchestration sessions.
    /// </summary>
    public Task<List<SessionSummary>> ListSessionsAsync(CancellationToken cancellationToken = default)
        => _sessionStore.ListSessionsAsync(cancellationToken);

    /// <summary>
    /// Gets resumable sessions.
    /// </summary>
    public Task<List<SessionSummary>> GetResumableSessionsAsync(CancellationToken cancellationToken = default)
        => _sessionStore.GetResumableSessionsAsync(cancellationToken);

    // ========================================
    // Workflow Execution
    // ========================================

    private async IAsyncEnumerable<OrchestrationEvent> ExecuteWorkflowAsync(
        OrchestrationSession session,
        HitlHandler? hitlHandler,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var context = session.Context;

        while (!context.CurrentState.IsTerminal())
        {
            cancellationToken.ThrowIfCancellationRequested();

            var previousState = context.CurrentState;
            var (events, terminalEvent, shouldBreak) = await ExecuteStateWithErrorHandlingAsync(
                session, hitlHandler, previousState, cancellationToken);

            // Yield collected events
            foreach (var evt in events)
            {
                yield return evt;
            }

            // Yield terminal event if any
            if (terminalEvent != null)
            {
                yield return terminalEvent;
            }

            if (shouldBreak)
            {
                yield break;
            }
        }

        // Completed
        if (context.CurrentState == OrchestrationState.Completed)
        {
            session.MarkCompleted();
            await _sessionStore.SaveSessionAsync(session, cancellationToken);

            yield return new OrchestrationCompletedEvent
            {
                SessionId = context.SessionId,
                TotalDuration = context.GetElapsedTime(),
                FinalMetrics = context.Training != null ? new ModelMetrics
                {
                    ModelName = context.Training.BestModelName,
                    PrimaryMetricName = context.Training.PrimaryMetricName,
                    PrimaryMetricValue = context.Training.PrimaryMetricValue,
                    AllMetrics = context.Training.Metrics
                } : null,
                Artifacts = context.Artifacts,
                Summary = BuildSummary(context)
            };
        }
    }

    /// <summary>
    /// Executes state with error handling, collecting events to avoid yield in try/catch.
    /// </summary>
    private async Task<(List<OrchestrationEvent> Events, OrchestrationEvent? TerminalEvent, bool ShouldBreak)>
        ExecuteStateWithErrorHandlingAsync(
            OrchestrationSession session,
            HitlHandler? hitlHandler,
            OrchestrationState previousState,
            CancellationToken cancellationToken)
    {
        var context = session.Context;
        var events = new List<OrchestrationEvent>();
        OrchestrationEvent? terminalEvent = null;
        var shouldBreak = false;

        try
        {
            // Execute current state
            await foreach (var evt in ExecuteStateAsync(session, hitlHandler, cancellationToken))
            {
                events.Add(evt);

                // Check for terminal events
                if (evt is OrchestrationFailedEvent or OrchestrationCancelledEvent)
                {
                    terminalEvent = evt;
                    shouldBreak = true;
                    return (events, terminalEvent, shouldBreak);
                }
            }

            // Move to next state if not paused
            if (session.Status != SessionStatus.Paused)
            {
                var nextState = context.CurrentState.GetNextState();
                if (nextState != context.CurrentState)
                {
                    session.RecordStateTransition(context.CurrentState, nextState);
                    context.CurrentState = nextState;

                    events.Add(new StateChangedEvent
                    {
                        SessionId = context.SessionId,
                        FromState = previousState,
                        ToState = nextState
                    });

                    await _sessionStore.SaveSessionAsync(session, cancellationToken);
                }
            }
        }
        catch (OperationCanceledException)
        {
            session.MarkCancelled("Cancelled by user");
            await _sessionStore.SaveSessionAsync(session, cancellationToken);

            terminalEvent = new OrchestrationCancelledEvent
            {
                SessionId = context.SessionId,
                Reason = "Cancelled by user",
                CancelledAtState = context.CurrentState,
                CanResume = false
            };
            shouldBreak = true;
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Error in state {State}", context.CurrentState);

            context.Errors.Add(new OrchestrationError
            {
                State = context.CurrentState,
                Message = ex.Message,
                Details = ex.ToString(),
                IsRecoverable = true
            });

            session.MarkFailed(ex.Message);
            await _sessionStore.SaveSessionAsync(session, cancellationToken);

            terminalEvent = new OrchestrationFailedEvent
            {
                SessionId = context.SessionId,
                Error = ex.Message,
                Details = ex.ToString(),
                FailedAtState = context.CurrentState,
                CanResume = true
            };
            shouldBreak = true;
        }

        return (events, terminalEvent, shouldBreak);
    }

    private async IAsyncEnumerable<OrchestrationEvent> ExecuteStateAsync(
        OrchestrationSession session,
        HitlHandler? hitlHandler,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var context = session.Context;
        var state = context.CurrentState;

        switch (state)
        {
            case OrchestrationState.NotStarted:
            case OrchestrationState.Initializing:
                yield return new PhaseStartedEvent
                {
                    SessionId = context.SessionId,
                    PhaseNumber = 0,
                    PhaseName = "Initialization",
                    Description = "Preparing orchestration environment"
                };
                // Initialize - just transition to next state
                break;

            case OrchestrationState.DataAnalysis:
                await foreach (var evt in ExecuteDataAnalysisPhaseAsync(session, cancellationToken))
                    yield return evt;
                break;

            case OrchestrationState.DataAnalysisReview:
            case OrchestrationState.ModelSelectionReview:
            case OrchestrationState.PreprocessingReview:
            case OrchestrationState.TrainingReview:
            case OrchestrationState.DeploymentReview:
                await foreach (var evt in ExecuteHitlCheckpointAsync(session, hitlHandler, cancellationToken))
                    yield return evt;
                break;

            case OrchestrationState.ModelRecommendation:
                await foreach (var evt in ExecuteModelRecommendationPhaseAsync(session, cancellationToken))
                    yield return evt;
                break;

            case OrchestrationState.Preprocessing:
                await foreach (var evt in ExecutePreprocessingPhaseAsync(session, cancellationToken))
                    yield return evt;
                break;

            case OrchestrationState.Training:
                await foreach (var evt in ExecuteTrainingPhaseAsync(session, cancellationToken))
                    yield return evt;
                break;

            case OrchestrationState.Evaluation:
                await foreach (var evt in ExecuteEvaluationPhaseAsync(session, cancellationToken))
                    yield return evt;
                break;

            case OrchestrationState.Deployment:
                await foreach (var evt in ExecuteDeploymentPhaseAsync(session, cancellationToken))
                    yield return evt;
                break;
        }
    }

    // ========================================
    // Phase Execution Methods
    // ========================================

    private async IAsyncEnumerable<OrchestrationEvent> ExecuteDataAnalysisPhaseAsync(
        OrchestrationSession session,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var context = session.Context;
        var startTime = DateTimeOffset.UtcNow;

        yield return new PhaseStartedEvent
        {
            SessionId = context.SessionId,
            PhaseNumber = 1,
            PhaseName = "Data Analysis",
            Description = "Analyzing dataset structure and quality"
        };

        yield return new AgentStartedEvent
        {
            SessionId = context.SessionId,
            AgentName = "DataAnalyst",
            AgentType = "DataAnalystAgent",
            TaskDescription = "Analyzing dataset"
        };

        context.DataAnalysis = await _agentCoordinator.ExecuteDataAnalysisAsync(context, cancellationToken);
        session.CreateCheckpoint("data-analysis-complete");
        await _sessionStore.SaveSessionAsync(session, cancellationToken);

        yield return new AgentCompletedEvent
        {
            SessionId = context.SessionId,
            AgentName = "DataAnalyst",
            AgentType = "DataAnalystAgent",
            Success = true,
            Duration = DateTimeOffset.UtcNow - startTime,
            Results = new Dictionary<string, object>
            {
                ["row_count"] = context.DataAnalysis.RowCount,
                ["column_count"] = context.DataAnalysis.ColumnCount,
                ["target"] = context.DataAnalysis.DetectedTargetColumn ?? "Unknown",
                ["task_type"] = context.DataAnalysis.InferredTaskType ?? "Unknown"
            }
        };

        yield return new PhaseCompletedEvent
        {
            SessionId = context.SessionId,
            PhaseNumber = 1,
            PhaseName = "Data Analysis",
            Duration = DateTimeOffset.UtcNow - startTime
        };
    }

    private async IAsyncEnumerable<OrchestrationEvent> ExecuteModelRecommendationPhaseAsync(
        OrchestrationSession session,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var context = session.Context;
        var startTime = DateTimeOffset.UtcNow;

        yield return new PhaseStartedEvent
        {
            SessionId = context.SessionId,
            PhaseNumber = 2,
            PhaseName = "Model Recommendation",
            Description = "Selecting optimal models and configuration"
        };

        yield return new AgentStartedEvent
        {
            SessionId = context.SessionId,
            AgentName = "ModelArchitect",
            AgentType = "ModelArchitectAgent",
            TaskDescription = "Recommending models"
        };

        context.ModelRecommendation = await _agentCoordinator.ExecuteModelRecommendationAsync(context, cancellationToken);
        session.CreateCheckpoint("model-recommendation-complete");
        await _sessionStore.SaveSessionAsync(session, cancellationToken);

        yield return new AgentCompletedEvent
        {
            SessionId = context.SessionId,
            AgentName = "ModelArchitect",
            AgentType = "ModelArchitectAgent",
            Success = true,
            Duration = DateTimeOffset.UtcNow - startTime,
            Results = new Dictionary<string, object>
            {
                ["task_type"] = context.ModelRecommendation.TaskType,
                ["metric"] = context.ModelRecommendation.PrimaryMetric,
                ["trainers"] = context.ModelRecommendation.RecommendedTrainers.Select(t => t.TrainerName).ToList()
            }
        };

        yield return new PhaseCompletedEvent
        {
            SessionId = context.SessionId,
            PhaseNumber = 2,
            PhaseName = "Model Recommendation",
            Duration = DateTimeOffset.UtcNow - startTime
        };
    }

    private async IAsyncEnumerable<OrchestrationEvent> ExecutePreprocessingPhaseAsync(
        OrchestrationSession session,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var context = session.Context;
        var startTime = DateTimeOffset.UtcNow;

        yield return new PhaseStartedEvent
        {
            SessionId = context.SessionId,
            PhaseNumber = 3,
            PhaseName = "Preprocessing",
            Description = "Transforming and preparing data for training"
        };

        var useIncremental = context.DataAnalysis!.RowCount >= _agentCoordinator.LargeDatasetThreshold;
        var agentType = useIncremental ? "IncrementalPreprocessingAgent" : "PreprocessingExpertAgent";

        yield return new AgentStartedEvent
        {
            SessionId = context.SessionId,
            AgentName = useIncremental ? "IncrementalPreprocessing" : "PreprocessingExpert",
            AgentType = agentType,
            TaskDescription = "Preprocessing data"
        };

        context.Preprocessing = await _agentCoordinator.ExecutePreprocessingAsync(context, cancellationToken);
        session.CreateCheckpoint("preprocessing-complete");
        await _sessionStore.SaveSessionAsync(session, cancellationToken);

        yield return new AgentCompletedEvent
        {
            SessionId = context.SessionId,
            AgentName = useIncremental ? "IncrementalPreprocessing" : "PreprocessingExpert",
            AgentType = agentType,
            Success = true,
            Duration = DateTimeOffset.UtcNow - startTime,
            Results = new Dictionary<string, object>
            {
                ["rows_before"] = context.Preprocessing.RowsBefore,
                ["rows_after"] = context.Preprocessing.RowsAfter,
                ["steps_count"] = context.Preprocessing.Steps.Count,
                ["incremental"] = useIncremental
            }
        };

        yield return new PhaseCompletedEvent
        {
            SessionId = context.SessionId,
            PhaseNumber = 3,
            PhaseName = "Preprocessing",
            Duration = DateTimeOffset.UtcNow - startTime
        };
    }

    private async IAsyncEnumerable<OrchestrationEvent> ExecuteTrainingPhaseAsync(
        OrchestrationSession session,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var context = session.Context;
        var startTime = DateTimeOffset.UtcNow;

        yield return new PhaseStartedEvent
        {
            SessionId = context.SessionId,
            PhaseNumber = 4,
            PhaseName = "Training",
            Description = "Training machine learning models"
        };

        yield return new ProgressUpdateEvent
        {
            SessionId = context.SessionId,
            Percentage = 70,
            CurrentOperation = "Executing AutoML training..."
        };

        // Get training configuration from context
        var metric = context.ModelRecommendation?.PrimaryMetric ?? "Accuracy";
        var maxTime = context.Options.MaxTrainingTimeSeconds ?? 300;

        // Use preprocessed data if available, otherwise use original data
        var trainingDataPath = context.Preprocessing?.OutputFilePath ?? context.DataFilePath;

        // Get project directory from data file path
        var projectPath = Path.GetDirectoryName(Path.GetFullPath(context.DataFilePath));

        // Build training config
        var trainingConfig = new MLoopTrainingConfig
        {
            TimeSeconds = maxTime,
            Metric = metric,
            TestSplit = 0.2,
            DataPath = trainingDataPath,
            ExperimentName = $"exp-{context.SessionId}"
        };

        _logger?.LogInformation("Starting training: DataPath={DataPath}, Metric={Metric}, Time={Time}s",
            trainingDataPath, metric, maxTime);

        // Execute actual training via MLoopProjectManager
        var trainingResult = await _projectManager.TrainModelAsync(trainingConfig, projectPath);

        if (!trainingResult.Success)
        {
            _logger?.LogError("Training failed: {Error}", trainingResult.Error ?? trainingResult.Output);

            context.Training = new TrainingResult
            {
                BestModelName = "None",
                PrimaryMetricName = metric,
                PrimaryMetricValue = 0,
                TrainingDuration = DateTimeOffset.UtcNow - startTime,
                ModelsEvaluated = 0,
                Confidence = 0
            };

            context.Errors.Add(new OrchestrationError
            {
                State = OrchestrationState.Training,
                Message = $"Training failed: {trainingResult.Error ?? "Unknown error"}",
                Details = trainingResult.Output,
                IsRecoverable = true
            });
        }
        else
        {
            // Parse training results
            var experimentId = trainingResult.Data.TryGetValue("ExperimentId", out var expId)
                ? expId.ToString()
                : null;
            var bestTrainer = trainingResult.Data.TryGetValue("BestTrainer", out var trainer)
                ? trainer.ToString() ?? "Unknown"
                : context.ModelRecommendation?.RecommendedTrainers.FirstOrDefault()?.TrainerName ?? "FastForest";
            var metricValue = trainingResult.Data.TryGetValue("MetricValue", out var mv) && mv is double d
                ? d
                : 0.0;

            _logger?.LogInformation("Training completed: ExperimentId={ExperimentId}, BestTrainer={Trainer}, Metric={MetricValue}",
                experimentId, bestTrainer, metricValue);

            context.Training = new TrainingResult
            {
                BestModelName = bestTrainer,
                PrimaryMetricName = metric,
                PrimaryMetricValue = metricValue,
                Metrics = new Dictionary<string, double>
                {
                    [metric] = metricValue
                },
                TrainingDuration = DateTimeOffset.UtcNow - startTime,
                ModelsEvaluated = 1, // MLoop AutoML evaluates multiple internally
                ExperimentId = experimentId,
                TopModels = [
                    new ModelSummary
                    {
                        ModelName = bestTrainer,
                        Score = metricValue,
                        TrainingTime = DateTimeOffset.UtcNow - startTime,
                        Rank = 1
                    }
                ],
                Confidence = metricValue > 0 ? 0.85 : 0.0
            };

            // Store experiment ID as artifact
            if (!string.IsNullOrEmpty(experimentId))
            {
                context.Artifacts["experiment_id"] = experimentId;
            }
        }

        session.CreateCheckpoint("training-complete");
        await _sessionStore.SaveSessionAsync(session, cancellationToken);

        yield return new PhaseCompletedEvent
        {
            SessionId = context.SessionId,
            PhaseNumber = 4,
            PhaseName = "Training",
            Duration = DateTimeOffset.UtcNow - startTime,
            Summary = new Dictionary<string, object>
            {
                ["best_model"] = context.Training.BestModelName,
                ["score"] = context.Training.PrimaryMetricValue,
                ["experiment_id"] = context.Training.ExperimentId ?? "N/A",
                ["success"] = trainingResult.Success
            }
        };
    }

    private async IAsyncEnumerable<OrchestrationEvent> ExecuteEvaluationPhaseAsync(
        OrchestrationSession session,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var context = session.Context;
        var startTime = DateTimeOffset.UtcNow;

        yield return new ProgressUpdateEvent
        {
            SessionId = context.SessionId,
            Percentage = 85,
            CurrentOperation = "Evaluating model on test set"
        };

        // TODO: Integrate with MLoop evaluation
        context.Evaluation = new EvaluationResult
        {
            TestMetrics = context.Training?.Metrics ?? [],
            Summary = "Model evaluation completed successfully",
            Confidence = 0.80
        };

        session.CreateCheckpoint("evaluation-complete");
        await _sessionStore.SaveSessionAsync(session, cancellationToken);

        await Task.CompletedTask; // Placeholder
    }

    private async IAsyncEnumerable<OrchestrationEvent> ExecuteDeploymentPhaseAsync(
        OrchestrationSession session,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var context = session.Context;

        yield return new PhaseStartedEvent
        {
            SessionId = context.SessionId,
            PhaseNumber = 5,
            PhaseName = "Deployment",
            Description = "Deploying trained model"
        };

        // TODO: Integrate with MLOpsManager
        context.Deployment = new DeploymentResult
        {
            Success = true,
            DeploymentTarget = "local",
            DeployedAt = DateTimeOffset.UtcNow,
            ModelVersion = "1.0.0"
        };

        session.CreateCheckpoint("deployment-complete");
        await _sessionStore.SaveSessionAsync(session, cancellationToken);

        yield return new PhaseCompletedEvent
        {
            SessionId = context.SessionId,
            PhaseNumber = 5,
            PhaseName = "Deployment",
            Duration = TimeSpan.FromSeconds(1)
        };
    }

    // ========================================
    // HITL Checkpoint Handling
    // ========================================

    private async IAsyncEnumerable<OrchestrationEvent> ExecuteHitlCheckpointAsync(
        OrchestrationSession session,
        HitlHandler? hitlHandler,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var context = session.Context;
        var state = context.CurrentState;
        var confidence = context.GetCurrentConfidence();

        // Check if HITL should be triggered
        if (!_hitlManager.ShouldTriggerHitl(state, confidence, context.Options))
        {
            // Auto-approve
            var autoDecision = new HitlDecision
            {
                CheckpointId = state.ToString(),
                State = state,
                SelectedOptionId = "approve",
                IsAutoApproval = true,
                Confidence = confidence,
                Timestamp = DateTimeOffset.UtcNow,
                ResponseTime = TimeSpan.Zero
            };
            context.HitlDecisions.Add(autoDecision);

            yield return new HitlResponseReceivedEvent
            {
                SessionId = context.SessionId,
                CheckpointId = state.ToString(),
                SelectedOptionId = "approve",
                IsAutoApproval = true,
                ResponseTime = TimeSpan.Zero
            };
            yield break;
        }

        // Create HITL request
        var request = _hitlManager.CreateHitlRequest(state, context);
        yield return request;

        // If no handler, pause the session
        if (hitlHandler == null)
        {
            session.MarkPaused();
            await _sessionStore.SaveSessionAsync(session, cancellationToken);
            _logger?.LogInformation("Session paused at HITL checkpoint: {State}", state);
            yield break;
        }

        // Wait for response
        var startTime = DateTimeOffset.UtcNow;
        var (optionId, comment) = await hitlHandler(request);
        var responseTime = DateTimeOffset.UtcNow - startTime;

        // Record decision
        var decision = new HitlDecision
        {
            CheckpointId = request.CheckpointId,
            State = state,
            SelectedOptionId = optionId,
            IsAutoApproval = false,
            Confidence = confidence,
            UserComment = comment,
            Timestamp = DateTimeOffset.UtcNow,
            ResponseTime = responseTime
        };
        context.HitlDecisions.Add(decision);
        await _sessionStore.SaveDecisionAsync(context.SessionId, decision, cancellationToken);

        yield return new HitlResponseReceivedEvent
        {
            SessionId = context.SessionId,
            CheckpointId = request.CheckpointId,
            SelectedOptionId = optionId,
            IsAutoApproval = false,
            UserComment = comment,
            ResponseTime = responseTime
        };

        // Process response
        var action = _hitlManager.ProcessResponse(state, optionId, context);

        if (action == HitlResponseAction.Cancel)
        {
            session.MarkCancelled("Cancelled by user at HITL checkpoint");
            await _sessionStore.SaveSessionAsync(session, cancellationToken);

            yield return new OrchestrationCancelledEvent
            {
                SessionId = context.SessionId,
                Reason = "Cancelled by user",
                CancelledAtState = state,
                CanResume = false
            };
        }

        // Other actions (Modify, Retry, Skip) would need additional handling
        // For now, just proceed to next state
    }

    // ========================================
    // Helpers
    // ========================================

    private static Dictionary<string, object> BuildSummary(OrchestrationContext context)
    {
        return new Dictionary<string, object>
        {
            ["session_id"] = context.SessionId,
            ["data_file"] = context.DataFilePath,
            ["total_duration"] = context.GetElapsedTime().ToString(@"hh\:mm\:ss"),
            ["rows_processed"] = context.DataAnalysis?.RowCount ?? 0,
            ["best_model"] = context.Training?.BestModelName ?? "N/A",
            ["final_score"] = context.Training?.PrimaryMetricValue ?? 0,
            ["hitl_decisions"] = context.HitlDecisions.Count,
            ["errors_count"] = context.Errors.Count
        };
    }
}
