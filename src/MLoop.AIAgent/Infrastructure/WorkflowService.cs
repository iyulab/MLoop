using Ironbees.AgentMode.Core.Workflow;
using Ironbees.AgentMode.Core.Workflow.Triggers;
using Ironbees.AgentMode.Models;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using System.Runtime.CompilerServices;

namespace MLoop.AIAgent.Infrastructure;

/// <summary>
/// MLoop workflow service using Ironbees YamlDrivenOrchestrator.
/// Provides YAML-based workflow execution for ML pipelines.
/// </summary>
public sealed class WorkflowService : IDisposable
{
    private readonly YamlDrivenOrchestrator _orchestrator;
    private readonly YamlWorkflowLoader _workflowLoader;
    private readonly ILogger<WorkflowService>? _logger;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the WorkflowService.
    /// </summary>
    /// <param name="agentsDirectory">Directory containing agent definitions.</param>
    /// <param name="orchestrator">The Ironbees orchestrator for agent access.</param>
    /// <param name="logger">Optional logger for diagnostics.</param>
    public WorkflowService(
        string agentsDirectory,
        IronbeesOrchestrator orchestrator,
        ILogger<WorkflowService>? logger = null)
    {
        ArgumentNullException.ThrowIfNull(orchestrator);
        ArgumentException.ThrowIfNullOrWhiteSpace(agentsDirectory);

        _logger = logger;
        _workflowLoader = new YamlWorkflowLoader();

        var triggerFactory = new TriggerEvaluatorFactory();
        var executorFactory = new MLoopAgentExecutorFactory(orchestrator);

        _orchestrator = new YamlDrivenOrchestrator(
            _workflowLoader,
            triggerFactory,
            executorFactory);

        _logger?.LogInformation("WorkflowService initialized with agents from: {AgentsDirectory}", agentsDirectory);
    }

    /// <summary>
    /// Loads a workflow definition from a YAML file.
    /// </summary>
    /// <param name="workflowPath">Path to the workflow YAML file.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The parsed workflow definition.</returns>
    public async Task<WorkflowDefinition> LoadWorkflowAsync(
        string workflowPath,
        CancellationToken cancellationToken = default)
    {
        _logger?.LogInformation("Loading workflow from: {WorkflowPath}", workflowPath);
        return await _workflowLoader.LoadFromFileAsync(workflowPath, cancellationToken);
    }

    /// <summary>
    /// Loads a workflow definition from YAML content string.
    /// </summary>
    /// <param name="yamlContent">YAML content string.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The parsed workflow definition.</returns>
    public async Task<WorkflowDefinition> LoadWorkflowFromStringAsync(
        string yamlContent,
        CancellationToken cancellationToken = default)
    {
        return await _workflowLoader.LoadFromStringAsync(yamlContent, cancellationToken);
    }

    /// <summary>
    /// Validates a workflow definition.
    /// </summary>
    /// <param name="workflow">Workflow definition to validate.</param>
    /// <returns>Validation result.</returns>
    public WorkflowValidationResult ValidateWorkflow(WorkflowDefinition workflow)
    {
        return _workflowLoader.Validate(workflow);
    }

    /// <summary>
    /// Executes a workflow and streams state updates.
    /// </summary>
    /// <param name="workflow">Workflow definition to execute.</param>
    /// <param name="input">Initial input for the workflow.</param>
    /// <param name="workingDirectory">Working directory for execution.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Async stream of workflow state updates.</returns>
    public async IAsyncEnumerable<WorkflowRuntimeState> ExecuteWorkflowAsync(
        WorkflowDefinition workflow,
        string input,
        string? workingDirectory = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var context = new WorkflowExecutionContext
        {
            WorkingDirectory = workingDirectory ?? Directory.GetCurrentDirectory()
        };

        _logger?.LogInformation(
            "Starting workflow execution: {WorkflowName} with input: {Input}",
            workflow.Name,
            input.Length > 100 ? input[..100] + "..." : input);

        await foreach (var state in _orchestrator.ExecuteAsync(workflow, input, context, cancellationToken))
        {
            _logger?.LogDebug(
                "Workflow state update - Execution: {ExecutionId}, State: {StateId}, Status: {Status}",
                state.ExecutionId,
                state.CurrentStateId,
                state.Status);

            yield return state;
        }
    }

    /// <summary>
    /// Executes a workflow from file path and streams state updates.
    /// </summary>
    /// <param name="workflowPath">Path to workflow YAML file.</param>
    /// <param name="input">Initial input for the workflow.</param>
    /// <param name="workingDirectory">Working directory for execution.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Async stream of workflow state updates.</returns>
    public async IAsyncEnumerable<WorkflowRuntimeState> ExecuteWorkflowFromFileAsync(
        string workflowPath,
        string input,
        string? workingDirectory = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var workflow = await LoadWorkflowAsync(workflowPath, cancellationToken);
        await foreach (var state in ExecuteWorkflowAsync(workflow, input, workingDirectory, cancellationToken))
        {
            yield return state;
        }
    }

    /// <summary>
    /// Approves a workflow waiting at a human gate.
    /// </summary>
    /// <param name="executionId">Workflow execution ID.</param>
    /// <param name="approved">Whether to approve.</param>
    /// <param name="feedback">Optional feedback for rejection.</param>
    public async Task ApproveAsync(string executionId, bool approved, string? feedback = null)
    {
        var decision = new ApprovalDecision
        {
            Approved = approved,
            Feedback = feedback
        };

        await _orchestrator.ApproveAsync(executionId, decision);
        _logger?.LogInformation(
            "Workflow {ExecutionId} {Decision} with feedback: {Feedback}",
            executionId,
            approved ? "approved" : "rejected",
            feedback ?? "none");
    }

    /// <summary>
    /// Cancels a running workflow execution.
    /// </summary>
    /// <param name="executionId">Workflow execution ID.</param>
    public async Task CancelAsync(string executionId)
    {
        await _orchestrator.CancelAsync(executionId);
        _logger?.LogInformation("Workflow {ExecutionId} cancelled", executionId);
    }

    /// <summary>
    /// Gets the current state of a workflow execution.
    /// </summary>
    /// <param name="executionId">Workflow execution ID.</param>
    /// <returns>Current workflow state.</returns>
    public async Task<WorkflowRuntimeState> GetStateAsync(string executionId)
    {
        return await _orchestrator.GetStateAsync(executionId);
    }

    /// <summary>
    /// Lists all active workflow executions.
    /// </summary>
    /// <returns>List of active execution summaries.</returns>
    public async Task<IReadOnlyList<WorkflowExecutionSummary>> ListActiveExecutionsAsync()
    {
        return await _orchestrator.ListActiveExecutionsAsync();
    }

    /// <summary>
    /// Disposes resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
    }
}

/// <summary>
/// MLoop-specific agent executor factory for workflow execution.
/// </summary>
internal sealed class MLoopAgentExecutorFactory : IAgentExecutorFactory
{
    private readonly IronbeesOrchestrator _orchestrator;

    public MLoopAgentExecutorFactory(IronbeesOrchestrator orchestrator)
    {
        _orchestrator = orchestrator ?? throw new ArgumentNullException(nameof(orchestrator));
    }

    public Task<IAgentExecutor> CreateExecutorAsync(
        string agentName,
        WorkflowExecutionContext context,
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult<IAgentExecutor>(new MLoopAgentExecutor(_orchestrator, agentName));
    }
}

/// <summary>
/// MLoop-specific agent executor that delegates to IronbeesOrchestrator.
/// </summary>
internal sealed class MLoopAgentExecutor : IAgentExecutor
{
    private readonly IronbeesOrchestrator _orchestrator;
    private readonly string _agentName;

    public MLoopAgentExecutor(IronbeesOrchestrator orchestrator, string agentName)
    {
        _orchestrator = orchestrator ?? throw new ArgumentNullException(nameof(orchestrator));
        _agentName = agentName ?? throw new ArgumentNullException(nameof(agentName));
    }

    public async Task<AgentExecutionResult> ExecuteAsync(
        string input,
        IReadOnlyDictionary<string, object> context,
        CancellationToken cancellationToken = default)
    {
        try
        {
            // Stream agent response and collect full result
            var responseBuilder = new System.Text.StringBuilder();
            await foreach (var chunk in _orchestrator.StreamAsync(input, _agentName, null, cancellationToken))
            {
                responseBuilder.Append(chunk);
            }

            var response = responseBuilder.ToString();

            return new AgentExecutionResult
            {
                Success = true,
                Data = new Dictionary<string, object>
                {
                    ["agent_response"] = response,
                    ["agent_name"] = _agentName
                }
            };
        }
        catch (Exception ex)
        {
            return new AgentExecutionResult
            {
                Success = false,
                ErrorMessage = ex.Message,
                Data = new Dictionary<string, object>
                {
                    ["agent_name"] = _agentName,
                    ["error_type"] = ex.GetType().Name
                }
            };
        }
    }
}
