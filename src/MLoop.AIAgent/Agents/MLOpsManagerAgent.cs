using Ironbees.AgentMode.Agents;
using Microsoft.Extensions.AI;
using MLoop.AIAgent.Core;
using MLoop.AIAgent.Core.Models;
using System.Text;

namespace MLoop.AIAgent.Agents;

/// <summary>
/// MLOps manager agent specializing in MLoop project lifecycle orchestration.
/// Executes training, evaluation, and prediction workflows using MLoopProjectManager.
/// </summary>
public class MLOpsManagerAgent : ConversationalAgent
{
    private readonly MLoopProjectManager _projectManager;

    private new const string SystemPrompt = @"# MLOps Manager Agent - System Prompt

You are an expert MLOps manager specializing in MLoop project lifecycle management. Your role is to orchestrate the entire ML workflow from project initialization to model deployment.

## Core Responsibilities

1. **Project Initialization**
   - Create MLoop projects with proper structure
   - Set up configuration files (mloop.yaml)
   - Initialize directory structure (.mloop/, experiments/, models/)

2. **Training Orchestration**
   - Execute model training with appropriate parameters
   - Monitor training progress and report status
   - Handle training failures with helpful error messages

3. **Model Evaluation**
   - Run model evaluation on test data
   - Report performance metrics clearly
   - Compare models across experiments

4. **Prediction Execution**
   - Execute batch predictions on new data
   - Handle single-instance predictions
   - Format prediction results appropriately

5. **Experiment Tracking**
   - Track experiment metadata and results
   - Organize models in experiment directories
   - Provide experiment history and comparison

## MLoop CLI Commands

You orchestrate these MLoop commands:

### mloop init
```bash
mloop init [project-name] --task [binary-classification|multiclass-classification|regression]
```
Creates new MLoop project with structure.

### mloop train
```bash
mloop train [data.csv] \
  --label [target-column] \
  --time [seconds] \
  --metric [accuracy|f1|auc|r2|rmse] \
  --test-split [0.0-1.0]
```
Trains model using AutoML.

### mloop evaluate
```bash
mloop evaluate [model.zip] [test-data.csv]
```
Evaluates model performance on test data.

### mloop predict
```bash
mloop predict [model.zip] [new-data.csv] -o [predictions.csv]
```
Generates predictions for new data.

## Workflow Orchestration

### Standard ML Workflow
1. **Initialize Project** → `mloop init`
2. **Analyze Data** → data-analyst agent
3. **Preprocess Data** → preprocessing-expert generates scripts
4. **Configure Training** → model-architect recommends settings
5. **Train Model** → `mloop train`
6. **Evaluate Model** → `mloop evaluate`
7. **Make Predictions** → `mloop predict`

### Error Handling
- Validate inputs before executing commands
- Provide clear error messages with solutions
- Suggest corrections for common mistakes
- Retry failed operations with adjusted parameters

## Communication Style

- **Status Updates**: Provide real-time progress reports
- **Clear Commands**: Show exact CLI commands being executed
- **Result Summaries**: Highlight key metrics and outcomes
- **Next Steps**: Suggest follow-up actions based on results

## Output Format

When executing MLoop operations:

```
🚀 Executing: [operation name]

**Command**:
```bash
[exact CLI command]
```

⏳ **Progress**:
[Real-time status updates]

✅ **Results**:

📊 **Performance Metrics**:
- Accuracy: [value]
- F1-Score: [value]
- AUC: [value]

💾 **Outputs**:
- Model: [path]
- Experiment: [experiment-id]
- Predictions: [path]

💡 **Next Steps**:
1. [Actionable recommendation]
2. [Another recommendation]
```

## Key Principles

1. **Reliable Execution**: Ensure commands execute successfully
2. **Clear Feedback**: Provide immediate, understandable status
3. **Error Recovery**: Handle failures gracefully with suggestions
4. **Best Practices**: Follow MLOps standards and conventions
5. **User Empowerment**: Teach users MLoop commands through examples

## Integration with MLoop

You are the **action executor** that:
- Translates agent recommendations into actual CLI commands
- Monitors execution and reports progress
- Manages the entire ML project lifecycle
- Coordinates with other agents for comprehensive workflow

### Integration Points

**From data-analyst**:
- Use data analysis results to validate training inputs
- Ensure data quality before training

**From preprocessing-expert**:
- Verify preprocessing scripts before training
- Coordinate preprocessing execution

**From model-architect**:
- Apply recommended AutoML configurations
- Use suggested metrics and parameters

## Command Execution Patterns

### Sequential Execution
For dependent operations:
```
1. Initialize project
2. Wait for completion
3. Run preprocessing
4. Wait for completion
5. Train model
6. Wait for completion
7. Evaluate model
```

### Parallel Execution
For independent operations:
```
- Evaluate multiple models simultaneously
- Generate predictions for different datasets
```

### Monitoring
- Parse command output for progress
- Extract metrics from training logs
- Report real-time status to user

## Error Scenarios

**Common Issues**:
- File not found → Verify paths and suggest corrections
- Invalid configuration → Explain requirements and fix
- Training failure → Analyze logs and suggest adjustments
- Out of memory → Recommend smaller time_limit or data subset

**Recovery Strategies**:
- Retry with adjusted parameters
- Suggest alternative approaches
- Provide diagnostic information
- Escalate to user for manual intervention

## Advanced Features

- **Experiment Comparison**: Compare results across multiple experiments
- **Model Selection**: Help choose best model based on metrics
- **Deployment Readiness**: Validate models before production
- **Performance Optimization**: Suggest improvements based on results

Always execute commands safely, validate inputs, and provide clear, actionable feedback to users.";

    /// <summary>
    /// Initializes a new instance of the MLOpsManagerAgent.
    /// </summary>
    /// <param name="chatClient">The chat client for LLM interactions with multi-provider support.</param>
    public MLOpsManagerAgent(IChatClient chatClient)
        : base(chatClient, SystemPrompt)
    {
        _projectManager = new MLoopProjectManager();
    }

    /// <summary>
    /// Initializes a new instance of the MLOpsManagerAgent with custom prompt.
    /// </summary>
    /// <param name="chatClient">The chat client for LLM interactions.</param>
    /// <param name="customSystemPrompt">Custom system prompt for specialized MLOps scenarios.</param>
    public MLOpsManagerAgent(IChatClient chatClient, string customSystemPrompt)
        : base(chatClient, customSystemPrompt)
    {
        _projectManager = new MLoopProjectManager();
    }

    /// <summary>
    /// Initializes a new instance of the MLOpsManagerAgent with custom project manager.
    /// </summary>
    /// <param name="chatClient">The chat client for LLM interactions.</param>
    /// <param name="projectManager">Custom MLoopProjectManager instance.</param>
    public MLOpsManagerAgent(IChatClient chatClient, MLoopProjectManager projectManager)
        : base(chatClient, SystemPrompt)
    {
        _projectManager = projectManager;
    }

    #region Project Initialization

    /// <summary>
    /// Initializes a new MLoop project and returns LLM-generated insights.
    /// </summary>
    /// <param name="config">Project configuration.</param>
    /// <param name="userQuery">Optional user query about the project setup.</param>
    /// <param name="options">Optional chat options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>LLM response with project initialization insights.</returns>
    public async Task<string> InitializeProjectAsync(
        MLoopProjectConfig config,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var result = await _projectManager.InitializeProjectAsync(config);
        var resultContext = FormatInitResultForLLM(result, config);

        var userMessage = string.IsNullOrWhiteSpace(userQuery)
            ? $"Please analyze this project initialization result and provide guidance:\n\n{resultContext}"
            : $"Based on this project initialization result:\n\n{resultContext}\n\nUser question: {userQuery}";

        return await RespondAsync(userMessage, options, cancellationToken);
    }

    /// <summary>
    /// Gets the raw project initialization result.
    /// </summary>
    public async Task<MLoopOperationResult> GetInitializeResultAsync(MLoopProjectConfig config)
    {
        return await _projectManager.InitializeProjectAsync(config);
    }

    #endregion

    #region Model Training

    /// <summary>
    /// Trains a model and returns LLM-generated insights.
    /// </summary>
    /// <param name="config">Training configuration.</param>
    /// <param name="projectPath">Optional project path.</param>
    /// <param name="userQuery">Optional user query about the training.</param>
    /// <param name="options">Optional chat options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>LLM response with training insights.</returns>
    public async Task<string> TrainModelAsync(
        MLoopTrainingConfig config,
        string? projectPath = null,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var result = await _projectManager.TrainModelAsync(config, projectPath);
        var resultContext = FormatTrainResultForLLM(result, config);

        var userMessage = string.IsNullOrWhiteSpace(userQuery)
            ? $"Please analyze this model training result and provide insights:\n\n{resultContext}"
            : $"Based on this model training result:\n\n{resultContext}\n\nUser question: {userQuery}";

        return await RespondAsync(userMessage, options, cancellationToken);
    }

    /// <summary>
    /// Gets the raw training result.
    /// </summary>
    public async Task<MLoopOperationResult> GetTrainResultAsync(
        MLoopTrainingConfig config,
        string? projectPath = null)
    {
        return await _projectManager.TrainModelAsync(config, projectPath);
    }

    #endregion

    #region Model Evaluation

    /// <summary>
    /// Evaluates a model and returns LLM-generated insights.
    /// </summary>
    /// <param name="experimentId">Optional experiment ID.</param>
    /// <param name="testDataPath">Optional test data path.</param>
    /// <param name="projectPath">Optional project path.</param>
    /// <param name="userQuery">Optional user query about the evaluation.</param>
    /// <param name="options">Optional chat options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>LLM response with evaluation insights.</returns>
    public async Task<string> EvaluateModelAsync(
        string? experimentId = null,
        string? testDataPath = null,
        string? projectPath = null,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var result = await _projectManager.EvaluateModelAsync(experimentId, testDataPath, projectPath);
        var resultContext = FormatEvaluateResultForLLM(result, experimentId);

        var userMessage = string.IsNullOrWhiteSpace(userQuery)
            ? $"Please analyze this model evaluation result and provide recommendations:\n\n{resultContext}"
            : $"Based on this model evaluation result:\n\n{resultContext}\n\nUser question: {userQuery}";

        return await RespondAsync(userMessage, options, cancellationToken);
    }

    /// <summary>
    /// Gets the raw evaluation result.
    /// </summary>
    public async Task<MLoopOperationResult> GetEvaluateResultAsync(
        string? experimentId = null,
        string? testDataPath = null,
        string? projectPath = null)
    {
        return await _projectManager.EvaluateModelAsync(experimentId, testDataPath, projectPath);
    }

    #endregion

    #region Predictions

    /// <summary>
    /// Makes predictions and returns LLM-generated insights.
    /// </summary>
    /// <param name="inputDataPath">Path to input data.</param>
    /// <param name="outputPath">Path for output predictions.</param>
    /// <param name="experimentId">Optional experiment ID.</param>
    /// <param name="projectPath">Optional project path.</param>
    /// <param name="userQuery">Optional user query about the predictions.</param>
    /// <param name="options">Optional chat options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>LLM response with prediction insights.</returns>
    public async Task<string> PredictAsync(
        string inputDataPath,
        string outputPath,
        string? experimentId = null,
        string? projectPath = null,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var result = await _projectManager.PredictAsync(inputDataPath, outputPath, experimentId, projectPath);
        var resultContext = FormatPredictResultForLLM(result, inputDataPath, outputPath);

        var userMessage = string.IsNullOrWhiteSpace(userQuery)
            ? $"Please analyze this prediction result and provide insights:\n\n{resultContext}"
            : $"Based on this prediction result:\n\n{resultContext}\n\nUser question: {userQuery}";

        return await RespondAsync(userMessage, options, cancellationToken);
    }

    /// <summary>
    /// Gets the raw prediction result.
    /// </summary>
    public async Task<MLoopOperationResult> GetPredictResultAsync(
        string inputDataPath,
        string outputPath,
        string? experimentId = null,
        string? projectPath = null)
    {
        return await _projectManager.PredictAsync(inputDataPath, outputPath, experimentId, projectPath);
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Executes preprocessing scripts and returns LLM-generated insights.
    /// </summary>
    /// <param name="projectPath">Path to the project.</param>
    /// <param name="userQuery">Optional user query about preprocessing.</param>
    /// <param name="options">Optional chat options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>LLM response with preprocessing insights.</returns>
    public async Task<string> PreprocessDataAsync(
        string projectPath,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var result = await _projectManager.PreprocessDataAsync(projectPath);
        var resultContext = FormatPreprocessResultForLLM(result);

        var userMessage = string.IsNullOrWhiteSpace(userQuery)
            ? $"Please analyze this preprocessing result and provide guidance:\n\n{resultContext}"
            : $"Based on this preprocessing result:\n\n{resultContext}\n\nUser question: {userQuery}";

        return await RespondAsync(userMessage, options, cancellationToken);
    }

    /// <summary>
    /// Gets the raw preprocessing result.
    /// </summary>
    public async Task<MLoopOperationResult> GetPreprocessResultAsync(string projectPath)
    {
        return await _projectManager.PreprocessDataAsync(projectPath);
    }

    #endregion

    #region Experiment Management

    /// <summary>
    /// Lists experiments and returns LLM-generated summary.
    /// </summary>
    /// <param name="projectPath">Optional project path.</param>
    /// <param name="userQuery">Optional user query about experiments.</param>
    /// <param name="options">Optional chat options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>LLM response with experiment summary.</returns>
    public async Task<string> ListExperimentsAsync(
        string? projectPath = null,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var experiments = await _projectManager.ListExperimentsAsync(projectPath);
        var resultContext = FormatExperimentsForLLM(experiments);

        var userMessage = string.IsNullOrWhiteSpace(userQuery)
            ? $"Please summarize these experiments and provide recommendations:\n\n{resultContext}"
            : $"Based on these experiments:\n\n{resultContext}\n\nUser question: {userQuery}";

        return await RespondAsync(userMessage, options, cancellationToken);
    }

    /// <summary>
    /// Gets the raw experiment list.
    /// </summary>
    public async Task<List<MLoopExperiment>> GetExperimentsAsync(string? projectPath = null)
    {
        return await _projectManager.ListExperimentsAsync(projectPath);
    }

    /// <summary>
    /// Promotes an experiment to production and returns LLM insights.
    /// </summary>
    /// <param name="experimentId">Experiment ID to promote.</param>
    /// <param name="projectPath">Optional project path.</param>
    /// <param name="userQuery">Optional user query.</param>
    /// <param name="options">Optional chat options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>LLM response about promotion.</returns>
    public async Task<string> PromoteExperimentAsync(
        string experimentId,
        string? projectPath = null,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var result = await _projectManager.PromoteExperimentAsync(experimentId, projectPath);
        var resultContext = FormatPromoteResultForLLM(result, experimentId);

        var userMessage = string.IsNullOrWhiteSpace(userQuery)
            ? $"Please analyze this promotion result:\n\n{resultContext}"
            : $"Based on this promotion result:\n\n{resultContext}\n\nUser question: {userQuery}";

        return await RespondAsync(userMessage, options, cancellationToken);
    }

    /// <summary>
    /// Gets the raw promotion result.
    /// </summary>
    public async Task<MLoopOperationResult> GetPromoteResultAsync(
        string experimentId,
        string? projectPath = null)
    {
        return await _projectManager.PromoteExperimentAsync(experimentId, projectPath);
    }

    #endregion

    #region Dataset Information

    /// <summary>
    /// Gets dataset information and returns LLM-generated summary.
    /// </summary>
    /// <param name="dataPath">Path to the dataset.</param>
    /// <param name="projectPath">Optional project path.</param>
    /// <param name="userQuery">Optional user query about the dataset.</param>
    /// <param name="options">Optional chat options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>LLM response with dataset information.</returns>
    public async Task<string> GetDatasetInfoAsync(
        string dataPath,
        string? projectPath = null,
        string? userQuery = null,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var result = await _projectManager.GetDatasetInfoAsync(dataPath, projectPath);
        var resultContext = FormatDatasetInfoForLLM(result, dataPath);

        var userMessage = string.IsNullOrWhiteSpace(userQuery)
            ? $"Please analyze this dataset information:\n\n{resultContext}"
            : $"Based on this dataset information:\n\n{resultContext}\n\nUser question: {userQuery}";

        return await RespondAsync(userMessage, options, cancellationToken);
    }

    /// <summary>
    /// Gets the raw dataset info result.
    /// </summary>
    public async Task<MLoopOperationResult> GetDatasetInfoResultAsync(
        string dataPath,
        string? projectPath = null)
    {
        return await _projectManager.GetDatasetInfoAsync(dataPath, projectPath);
    }

    #endregion

    #region Result Formatting

    private static string FormatInitResultForLLM(MLoopOperationResult result, MLoopProjectConfig config)
    {
        var sb = new StringBuilder();

        sb.AppendLine("## Project Initialization Result");
        sb.AppendLine();
        sb.AppendLine($"- **Project Name**: {config.ProjectName}");
        sb.AppendLine($"- **Task Type**: {config.TaskType}");
        sb.AppendLine($"- **Data Path**: {config.DataPath}");
        sb.AppendLine($"- **Label Column**: {config.LabelColumn}");
        sb.AppendLine();
        sb.AppendLine($"### Execution Status");
        sb.AppendLine($"- **Success**: {(result.Success ? "✅ Yes" : "❌ No")}");
        sb.AppendLine($"- **Exit Code**: {result.ExitCode}");

        if (result.Data.TryGetValue("ProjectDirectory", out var projectDir))
        {
            sb.AppendLine($"- **Project Directory**: {projectDir}");
        }

        if (!string.IsNullOrWhiteSpace(result.Output))
        {
            sb.AppendLine();
            sb.AppendLine("### Output");
            sb.AppendLine("```");
            sb.AppendLine(result.Output);
            sb.AppendLine("```");
        }

        if (!string.IsNullOrWhiteSpace(result.Error))
        {
            sb.AppendLine();
            sb.AppendLine("### Errors");
            sb.AppendLine("```");
            sb.AppendLine(result.Error);
            sb.AppendLine("```");
        }

        return sb.ToString();
    }

    private static string FormatTrainResultForLLM(MLoopOperationResult result, MLoopTrainingConfig config)
    {
        var sb = new StringBuilder();

        sb.AppendLine("## Model Training Result");
        sb.AppendLine();
        sb.AppendLine("### Configuration");
        sb.AppendLine($"- **Time Limit**: {config.TimeSeconds} seconds");
        sb.AppendLine($"- **Metric**: {config.Metric}");
        sb.AppendLine($"- **Test Split**: {config.TestSplit:P0}");

        if (!string.IsNullOrEmpty(config.ExperimentName))
        {
            sb.AppendLine($"- **Experiment Name**: {config.ExperimentName}");
        }

        sb.AppendLine();
        sb.AppendLine("### Execution Status");
        sb.AppendLine($"- **Success**: {(result.Success ? "✅ Yes" : "❌ No")}");
        sb.AppendLine($"- **Exit Code**: {result.ExitCode}");

        // Extract metrics from result data
        if (result.Data.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("### Training Results");

            if (result.Data.TryGetValue("ExperimentId", out var expId))
            {
                sb.AppendLine($"- **Experiment ID**: {expId}");
            }

            if (result.Data.TryGetValue("BestTrainer", out var trainer))
            {
                sb.AppendLine($"- **Best Trainer**: {trainer}");
            }

            if (result.Data.TryGetValue("MetricValue", out var metricVal))
            {
                sb.AppendLine($"- **Metric Value**: {metricVal}");
            }
        }

        if (!string.IsNullOrWhiteSpace(result.Output))
        {
            sb.AppendLine();
            sb.AppendLine("### Output");
            sb.AppendLine("```");
            sb.AppendLine(result.Output);
            sb.AppendLine("```");
        }

        if (!string.IsNullOrWhiteSpace(result.Error))
        {
            sb.AppendLine();
            sb.AppendLine("### Errors");
            sb.AppendLine("```");
            sb.AppendLine(result.Error);
            sb.AppendLine("```");
        }

        return sb.ToString();
    }

    private static string FormatEvaluateResultForLLM(MLoopOperationResult result, string? experimentId)
    {
        var sb = new StringBuilder();

        sb.AppendLine("## Model Evaluation Result");
        sb.AppendLine();

        if (!string.IsNullOrEmpty(experimentId))
        {
            sb.AppendLine($"- **Experiment ID**: {experimentId}");
        }

        sb.AppendLine($"- **Success**: {(result.Success ? "✅ Yes" : "❌ No")}");
        sb.AppendLine($"- **Exit Code**: {result.ExitCode}");

        // Extract evaluation metrics
        if (result.Data.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("### Evaluation Metrics");

            foreach (var (key, value) in result.Data)
            {
                sb.AppendLine($"- **{key}**: {value}");
            }
        }

        if (!string.IsNullOrWhiteSpace(result.Output))
        {
            sb.AppendLine();
            sb.AppendLine("### Output");
            sb.AppendLine("```");
            sb.AppendLine(result.Output);
            sb.AppendLine("```");
        }

        if (!string.IsNullOrWhiteSpace(result.Error))
        {
            sb.AppendLine();
            sb.AppendLine("### Errors");
            sb.AppendLine("```");
            sb.AppendLine(result.Error);
            sb.AppendLine("```");
        }

        return sb.ToString();
    }

    private static string FormatPredictResultForLLM(
        MLoopOperationResult result,
        string inputPath,
        string outputPath)
    {
        var sb = new StringBuilder();

        sb.AppendLine("## Prediction Result");
        sb.AppendLine();
        sb.AppendLine($"- **Input Data**: {inputPath}");
        sb.AppendLine($"- **Output Path**: {outputPath}");
        sb.AppendLine($"- **Success**: {(result.Success ? "✅ Yes" : "❌ No")}");
        sb.AppendLine($"- **Exit Code**: {result.ExitCode}");

        if (!string.IsNullOrWhiteSpace(result.Output))
        {
            sb.AppendLine();
            sb.AppendLine("### Output");
            sb.AppendLine("```");
            sb.AppendLine(result.Output);
            sb.AppendLine("```");
        }

        if (!string.IsNullOrWhiteSpace(result.Error))
        {
            sb.AppendLine();
            sb.AppendLine("### Errors");
            sb.AppendLine("```");
            sb.AppendLine(result.Error);
            sb.AppendLine("```");
        }

        return sb.ToString();
    }

    private static string FormatPreprocessResultForLLM(MLoopOperationResult result)
    {
        var sb = new StringBuilder();

        sb.AppendLine("## Preprocessing Result");
        sb.AppendLine();
        sb.AppendLine($"- **Success**: {(result.Success ? "✅ Yes" : "❌ No")}");
        sb.AppendLine($"- **Exit Code**: {result.ExitCode}");

        if (!string.IsNullOrWhiteSpace(result.Output))
        {
            sb.AppendLine();
            sb.AppendLine("### Output");
            sb.AppendLine("```");
            sb.AppendLine(result.Output);
            sb.AppendLine("```");
        }

        if (!string.IsNullOrWhiteSpace(result.Error))
        {
            sb.AppendLine();
            sb.AppendLine("### Errors");
            sb.AppendLine("```");
            sb.AppendLine(result.Error);
            sb.AppendLine("```");
        }

        return sb.ToString();
    }

    private static string FormatExperimentsForLLM(List<MLoopExperiment> experiments)
    {
        var sb = new StringBuilder();

        sb.AppendLine("## Experiments");
        sb.AppendLine();

        if (experiments.Count == 0)
        {
            sb.AppendLine("No experiments found in this project.");
            return sb.ToString();
        }

        sb.AppendLine($"Total experiments: {experiments.Count}");
        sb.AppendLine();

        sb.AppendLine("| ID | Name | Trainer | Metric | Value | Date | Production |");
        sb.AppendLine("|-----|------|---------|--------|-------|------|------------|");

        foreach (var exp in experiments.OrderByDescending(e => e.Timestamp))
        {
            var prod = exp.IsProduction ? "✅" : "";
            sb.AppendLine($"| {exp.Id[..8]}... | {exp.Name} | {exp.Trainer} | {exp.MetricName} | {exp.MetricValue:F4} | {exp.Timestamp:yyyy-MM-dd} | {prod} |");
        }

        // Highlight best experiment
        var best = experiments.OrderByDescending(e => e.MetricValue).FirstOrDefault();
        if (best != null)
        {
            sb.AppendLine();
            sb.AppendLine($"**Best Experiment**: {best.Name} ({best.Trainer}) - {best.MetricName}: {best.MetricValue:F4}");
        }

        return sb.ToString();
    }

    private static string FormatPromoteResultForLLM(MLoopOperationResult result, string experimentId)
    {
        var sb = new StringBuilder();

        sb.AppendLine("## Promotion Result");
        sb.AppendLine();
        sb.AppendLine($"- **Experiment ID**: {experimentId}");
        sb.AppendLine($"- **Success**: {(result.Success ? "✅ Yes" : "❌ No")}");
        sb.AppendLine($"- **Exit Code**: {result.ExitCode}");

        if (!string.IsNullOrWhiteSpace(result.Output))
        {
            sb.AppendLine();
            sb.AppendLine("### Output");
            sb.AppendLine("```");
            sb.AppendLine(result.Output);
            sb.AppendLine("```");
        }

        if (!string.IsNullOrWhiteSpace(result.Error))
        {
            sb.AppendLine();
            sb.AppendLine("### Errors");
            sb.AppendLine("```");
            sb.AppendLine(result.Error);
            sb.AppendLine("```");
        }

        return sb.ToString();
    }

    private static string FormatDatasetInfoForLLM(MLoopOperationResult result, string dataPath)
    {
        var sb = new StringBuilder();

        sb.AppendLine("## Dataset Information");
        sb.AppendLine();
        sb.AppendLine($"- **Data Path**: {dataPath}");
        sb.AppendLine($"- **Success**: {(result.Success ? "✅ Yes" : "❌ No")}");
        sb.AppendLine($"- **Exit Code**: {result.ExitCode}");

        if (!string.IsNullOrWhiteSpace(result.Output))
        {
            sb.AppendLine();
            sb.AppendLine("### Details");
            sb.AppendLine("```");
            sb.AppendLine(result.Output);
            sb.AppendLine("```");
        }

        if (!string.IsNullOrWhiteSpace(result.Error))
        {
            sb.AppendLine();
            sb.AppendLine("### Errors");
            sb.AppendLine("```");
            sb.AppendLine(result.Error);
            sb.AppendLine("```");
        }

        return sb.ToString();
    }

    #endregion
}
