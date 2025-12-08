using Ironbees.AgentMode.Agents;
using Microsoft.Extensions.AI;
using MLoop.AIAgent.Core;
using MLoop.AIAgent.Core.Models;
using MLoop.AIAgent.Tools;
using System.Text;

namespace MLoop.AIAgent.Agents;

/// <summary>
/// MLOps manager agent specializing in MLoop project lifecycle orchestration.
/// Supports LLM tool use via MS.Extensions.AI for autonomous ML workflow execution.
/// </summary>
public class MLOpsManagerAgent : ConversationalAgent
{
    private readonly MLoopProjectManager _projectManager;
    private readonly MLOpsTools _tools;
    private readonly IList<AITool> _aiTools;

    private new const string SystemPrompt = @"# MLOps Manager Agent - System Prompt

You are an expert MLOps manager specializing in MLoop project lifecycle management. You have direct access to ML tools that you can invoke to execute operations.

## Available Tools

You have access to these tools for ML operations:

1. **initialize_project** - Create new MLoop projects
2. **train_model** - Train ML models using AutoML
3. **evaluate_model** - Evaluate model performance
4. **predict** - Make predictions with trained models
5. **list_experiments** - View experiment history
6. **promote_experiment** - Deploy models to production
7. **get_dataset_info** - Analyze dataset statistics
8. **preprocess_data** - Run preprocessing pipelines

## Tool Usage Guidelines

When a user requests an ML operation:
1. **Analyze** the request to understand requirements
2. **Invoke** the appropriate tool with correct parameters
3. **Interpret** tool results and explain to the user
4. **Recommend** next steps based on results

## Standard ML Workflow

For complete ML pipelines:
1. `initialize_project` - Set up project structure
2. `get_dataset_info` - Understand the data
3. `preprocess_data` - Clean and prepare data
4. `train_model` - Train with AutoML
5. `evaluate_model` - Assess performance
6. `promote_experiment` - Deploy best model
7. `predict` - Generate predictions

## Response Format

When executing tools:
- Report which tool you're invoking
- Present results clearly with key metrics highlighted
- Provide actionable insights and recommendations
- Suggest logical next steps

## Error Handling

If a tool fails:
- Explain the error clearly
- Suggest parameter corrections
- Propose alternative approaches
- Guide the user to resolution

## Key Principles

1. **Proactive Execution**: Invoke tools when appropriate without asking
2. **Clear Communication**: Explain what you're doing and why
3. **Result Interpretation**: Don't just show results - explain them
4. **Continuous Guidance**: Always suggest next steps";

    /// <summary>
    /// Initializes a new instance of the MLOpsManagerAgent.
    /// </summary>
    /// <param name="chatClient">The chat client for LLM interactions.</param>
    public MLOpsManagerAgent(IChatClient chatClient)
        : base(chatClient, SystemPrompt)
    {
        _projectManager = new MLoopProjectManager();
        _tools = new MLOpsTools(_projectManager);
        _aiTools = _tools.CreateTools();
    }

    /// <summary>
    /// Initializes a new instance of the MLOpsManagerAgent with custom project path.
    /// </summary>
    /// <param name="chatClient">The chat client for LLM interactions.</param>
    /// <param name="projectPath">Default project path for operations.</param>
    public MLOpsManagerAgent(IChatClient chatClient, string projectPath)
        : base(chatClient, SystemPrompt)
    {
        _projectManager = new MLoopProjectManager();
        _tools = new MLOpsTools(_projectManager, projectPath);
        _aiTools = _tools.CreateTools();
    }

    /// <summary>
    /// Initializes a new instance with custom project manager.
    /// </summary>
    /// <param name="chatClient">The chat client for LLM interactions.</param>
    /// <param name="projectManager">Custom MLoopProjectManager instance.</param>
    /// <param name="projectPath">Optional default project path.</param>
    public MLOpsManagerAgent(
        IChatClient chatClient,
        MLoopProjectManager projectManager,
        string? projectPath = null)
        : base(chatClient, SystemPrompt)
    {
        _projectManager = projectManager;
        _tools = new MLOpsTools(_projectManager, projectPath);
        _aiTools = _tools.CreateTools();
    }

    #region Tool-Enabled Responses

    /// <summary>
    /// Process a request with tool use enabled.
    /// The LLM can autonomously invoke MLOps tools to fulfill the request.
    /// </summary>
    /// <param name="userMessage">User's message or request.</param>
    /// <param name="options">Optional chat options (tools will be added).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>LLM response after tool execution.</returns>
    public async Task<string> ProcessWithToolsAsync(
        string userMessage,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new ChatOptions();
        options.Tools = _aiTools;

        return await RespondAsync(userMessage, options, cancellationToken);
    }

    /// <summary>
    /// Stream a response with tool use enabled.
    /// </summary>
    /// <param name="userMessage">User's message or request.</param>
    /// <param name="options">Optional chat options (tools will be added).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Async enumerable of response chunks.</returns>
    public async IAsyncEnumerable<string> StreamWithToolsAsync(
        string userMessage,
        ChatOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= new ChatOptions();
        options.Tools = _aiTools;

        await foreach (var chunk in StreamResponseAsync(userMessage, options, cancellationToken))
        {
            yield return chunk;
        }
    }

    /// <summary>
    /// Gets the list of available tools for inspection.
    /// </summary>
    public IReadOnlyList<AITool> AvailableTools => _aiTools.AsReadOnly();

    #endregion

    #region Direct Tool Access (Legacy Compatibility)

    /// <summary>
    /// Direct access to train model without LLM involvement.
    /// </summary>
    public async Task<MLoopOperationResult> TrainModelDirectAsync(
        MLoopTrainingConfig config,
        string? projectPath = null)
    {
        return await _projectManager.TrainModelAsync(config, projectPath);
    }

    /// <summary>
    /// Direct access to evaluate model without LLM involvement.
    /// </summary>
    public async Task<MLoopOperationResult> EvaluateModelDirectAsync(
        string? experimentId = null,
        string? testDataPath = null,
        string? projectPath = null)
    {
        return await _projectManager.EvaluateModelAsync(experimentId, testDataPath, projectPath);
    }

    /// <summary>
    /// Direct access to make predictions without LLM involvement.
    /// </summary>
    public async Task<MLoopOperationResult> PredictDirectAsync(
        string inputDataPath,
        string outputPath,
        string? experimentId = null,
        string? projectPath = null)
    {
        return await _projectManager.PredictAsync(inputDataPath, outputPath, experimentId, projectPath);
    }

    /// <summary>
    /// Direct access to list experiments without LLM involvement.
    /// </summary>
    public async Task<List<MLoopExperiment>> ListExperimentsDirectAsync(string? projectPath = null)
    {
        return await _projectManager.ListExperimentsAsync(projectPath);
    }

    /// <summary>
    /// Direct access to initialize project without LLM involvement.
    /// </summary>
    public async Task<MLoopOperationResult> InitializeProjectDirectAsync(MLoopProjectConfig config)
    {
        return await _projectManager.InitializeProjectAsync(config);
    }

    /// <summary>
    /// Direct access to preprocess data without LLM involvement.
    /// </summary>
    public async Task<MLoopOperationResult> PreprocessDataDirectAsync(string projectPath)
    {
        return await _projectManager.PreprocessDataAsync(projectPath);
    }

    /// <summary>
    /// Direct access to get dataset info without LLM involvement.
    /// </summary>
    public async Task<MLoopOperationResult> GetDatasetInfoDirectAsync(
        string dataPath,
        string? projectPath = null)
    {
        return await _projectManager.GetDatasetInfoAsync(dataPath, projectPath);
    }

    /// <summary>
    /// Direct access to promote experiment without LLM involvement.
    /// </summary>
    public async Task<MLoopOperationResult> PromoteExperimentDirectAsync(
        string experimentId,
        string? projectPath = null)
    {
        return await _projectManager.PromoteExperimentAsync(experimentId, projectPath);
    }

    #endregion

    #region Guided Operations (LLM + Tools)

    /// <summary>
    /// Initialize a project with LLM guidance and tool execution.
    /// </summary>
    public async Task<string> InitializeProjectAsync(
        MLoopProjectConfig config,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var request = $"Initialize a new MLoop project with these settings:\n" +
                      $"- Project Name: {config.ProjectName}\n" +
                      $"- Task Type: {config.TaskType}\n" +
                      $"- Data Path: {config.DataPath}\n" +
                      $"- Label Column: {config.LabelColumn}";

        if (!string.IsNullOrWhiteSpace(userQuery))
        {
            request += $"\n\nAdditional context: {userQuery}";
        }

        return await ProcessWithToolsAsync(request, options, cancellationToken);
    }

    /// <summary>
    /// Train a model with LLM guidance and tool execution.
    /// </summary>
    public async Task<string> TrainModelAsync(
        MLoopTrainingConfig config,
        string? projectPath = null,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var request = $"Train a model with these settings:\n" +
                      $"- Time Limit: {config.TimeSeconds} seconds\n" +
                      $"- Metric: {config.Metric}\n" +
                      $"- Test Split: {config.TestSplit:P0}";

        if (!string.IsNullOrEmpty(config.DataPath))
        {
            request += $"\n- Data Path: {config.DataPath}";
        }

        if (!string.IsNullOrEmpty(config.ExperimentName))
        {
            request += $"\n- Experiment Name: {config.ExperimentName}";
        }

        if (!string.IsNullOrWhiteSpace(userQuery))
        {
            request += $"\n\nAdditional context: {userQuery}";
        }

        return await ProcessWithToolsAsync(request, options, cancellationToken);
    }

    /// <summary>
    /// Evaluate a model with LLM guidance and tool execution.
    /// </summary>
    public async Task<string> EvaluateModelAsync(
        string? experimentId = null,
        string? testDataPath = null,
        string? projectPath = null,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var request = "Evaluate the model";

        if (!string.IsNullOrEmpty(experimentId))
        {
            request += $" from experiment {experimentId}";
        }

        if (!string.IsNullOrEmpty(testDataPath))
        {
            request += $" using test data at {testDataPath}";
        }

        request += ". Analyze the results and provide recommendations.";

        if (!string.IsNullOrWhiteSpace(userQuery))
        {
            request += $"\n\nAdditional context: {userQuery}";
        }

        return await ProcessWithToolsAsync(request, options, cancellationToken);
    }

    /// <summary>
    /// Make predictions with LLM guidance and tool execution.
    /// </summary>
    public async Task<string> PredictAsync(
        string inputDataPath,
        string outputPath,
        string? experimentId = null,
        string? projectPath = null,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var request = $"Make predictions using input data from {inputDataPath} and save results to {outputPath}";

        if (!string.IsNullOrEmpty(experimentId))
        {
            request += $" using model from experiment {experimentId}";
        }

        request += ".";

        if (!string.IsNullOrWhiteSpace(userQuery))
        {
            request += $"\n\nAdditional context: {userQuery}";
        }

        return await ProcessWithToolsAsync(request, options, cancellationToken);
    }

    /// <summary>
    /// List and analyze experiments with LLM guidance.
    /// </summary>
    public async Task<string> ListExperimentsAsync(
        string? projectPath = null,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var request = "List all experiments and provide a summary with recommendations for the best model to use.";

        if (!string.IsNullOrWhiteSpace(userQuery))
        {
            request += $"\n\nAdditional context: {userQuery}";
        }

        return await ProcessWithToolsAsync(request, options, cancellationToken);
    }

    /// <summary>
    /// Promote an experiment with LLM guidance.
    /// </summary>
    public async Task<string> PromoteExperimentAsync(
        string experimentId,
        string? projectPath = null,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var request = $"Promote experiment {experimentId} to production and explain the implications.";

        if (!string.IsNullOrWhiteSpace(userQuery))
        {
            request += $"\n\nAdditional context: {userQuery}";
        }

        return await ProcessWithToolsAsync(request, options, cancellationToken);
    }

    /// <summary>
    /// Get dataset information with LLM analysis.
    /// </summary>
    public async Task<string> GetDatasetInfoAsync(
        string dataPath,
        string? projectPath = null,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var request = $"Analyze the dataset at {dataPath} and provide insights about data quality, feature characteristics, and preparation recommendations.";

        if (!string.IsNullOrWhiteSpace(userQuery))
        {
            request += $"\n\nAdditional context: {userQuery}";
        }

        return await ProcessWithToolsAsync(request, options, cancellationToken);
    }

    /// <summary>
    /// Execute preprocessing with LLM guidance.
    /// </summary>
    public async Task<string> PreprocessDataAsync(
        string projectPath,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var request = "Execute the preprocessing pipeline and report on data transformations applied.";

        if (!string.IsNullOrWhiteSpace(userQuery))
        {
            request += $"\n\nAdditional context: {userQuery}";
        }

        return await ProcessWithToolsAsync(request, options, cancellationToken);
    }

    #endregion

    #region Workflow Orchestration

    /// <summary>
    /// Execute a complete ML pipeline with LLM orchestration.
    /// </summary>
    /// <param name="config">Training configuration.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Pipeline execution summary.</returns>
    public async Task<string> ExecuteFullPipelineAsync(
        MLoopTrainingConfig config,
        CancellationToken cancellationToken = default)
    {
        var dataPath = config.DataPath ?? "project default data";
        var request = $@"Execute a complete ML pipeline for this task:

1. First, analyze the dataset at {dataPath}
2. Then, train a model using the project's configured label column
3. Use metric: {config.Metric}, time limit: {config.TimeSeconds}s
4. Evaluate the trained model
5. If performance is acceptable, recommend promotion to production

Execute these steps sequentially using the available tools and provide a comprehensive summary at the end.";

        return await ProcessWithToolsAsync(request, cancellationToken: cancellationToken);
    }

    #endregion
}
