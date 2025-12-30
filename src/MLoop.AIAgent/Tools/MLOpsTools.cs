using System.ComponentModel;
using System.Text.Json;
using Microsoft.Extensions.AI;
using MLoop.AIAgent.Core;
using MLoop.AIAgent.Core.Models;

namespace MLoop.AIAgent.Tools;

/// <summary>
/// MLOps tools for LLM invocation via MS.Extensions.AI AIFunctionFactory.
/// Implements Ironbees Thin Wrapper philosophy - tool execution is MS.Extensions.AI's responsibility.
/// </summary>
public class MLOpsTools
{
    private readonly MLoopProjectManager _projectManager;
    private readonly string? _defaultProjectPath;

    /// <summary>
    /// Initializes a new instance of MLOpsTools.
    /// </summary>
    /// <param name="projectManager">The MLoopProjectManager instance.</param>
    /// <param name="defaultProjectPath">Optional default project path.</param>
    public MLOpsTools(MLoopProjectManager projectManager, string? defaultProjectPath = null)
    {
        _projectManager = projectManager ?? throw new ArgumentNullException(nameof(projectManager));
        _defaultProjectPath = defaultProjectPath;
    }

    /// <summary>
    /// Creates all MLOps tools for LLM invocation.
    /// </summary>
    /// <returns>Array of AITools for ChatOptions.Tools.</returns>
    public IList<AITool> CreateTools()
    {
        return
        [
            // Project Management
            AIFunctionFactory.Create(InitializeProjectAsync,
                name: "initialize_project",
                description: "Initialize a new MLoop ML project with the specified configuration"),

            // Training
            AIFunctionFactory.Create(TrainModelAsync,
                name: "train_model",
                description: "Train an ML model using AutoML with the specified data and parameters"),

            // Evaluation
            AIFunctionFactory.Create(EvaluateModelAsync,
                name: "evaluate_model",
                description: "Evaluate a trained model's performance on test data"),

            // Prediction
            AIFunctionFactory.Create(PredictAsync,
                name: "predict",
                description: "Make predictions using a trained model on new data"),

            // Experiment Management
            AIFunctionFactory.Create(ListExperimentsAsync,
                name: "list_experiments",
                description: "List all experiments in the project with their metrics and status"),

            AIFunctionFactory.Create(PromoteExperimentAsync,
                name: "promote_experiment",
                description: "Promote an experiment to production status"),

            // Data Operations
            AIFunctionFactory.Create(GetDatasetInfoAsync,
                name: "get_dataset_info",
                description: "Get detailed information about a dataset including columns, types, and statistics"),

            AIFunctionFactory.Create(PreprocessDataAsync,
                name: "preprocess_data",
                description: "Execute preprocessing scripts on the project data")
        ];
    }

    #region Tool Implementations

    /// <summary>
    /// Initialize a new MLoop project.
    /// </summary>
    [Description("Initialize a new MLoop ML project")]
    public async Task<string> InitializeProjectAsync(
        [Description("Name of the project")] string projectName,
        [Description("ML task type: binary-classification, multiclass-classification, or regression")] string taskType,
        [Description("Path to the training data CSV file")] string dataPath,
        [Description("Name of the target/label column")] string labelColumn,
        [Description("Directory where the project will be created (optional)")] string? projectDirectory = null)
    {
        var config = new MLoopProjectConfig
        {
            ProjectName = projectName,
            TaskType = taskType,
            DataPath = dataPath,
            LabelColumn = labelColumn,
            ProjectDirectory = projectDirectory
        };

        var result = await _projectManager.InitializeProjectAsync(config);
        return FormatOperationResult("Project Initialization", result, new
        {
            projectName,
            taskType,
            dataPath,
            labelColumn
        });
    }

    /// <summary>
    /// Train an ML model using AutoML.
    /// Note: Label column is determined from project configuration set during initialization.
    /// </summary>
    [Description("Train an ML model using AutoML. Label column uses project configuration.")]
    public async Task<string> TrainModelAsync(
        [Description("Training time limit in seconds (default: 120)")] int timeSeconds = 120,
        [Description("Optimization metric: accuracy, f1, auc, r2, rmse")] string metric = "accuracy",
        [Description("Fraction of data to use for testing (0.0-1.0, default: 0.2)")] double testSplit = 0.2,
        [Description("Path to training data CSV file (optional, uses project data)")] string? dataPath = null,
        [Description("Optional experiment name for tracking")] string? experimentName = null)
    {
        var config = new MLoopTrainingConfig
        {
            DataPath = dataPath,
            TimeSeconds = timeSeconds,
            Metric = metric,
            TestSplit = testSplit,
            ExperimentName = experimentName
        };

        var result = await _projectManager.TrainModelAsync(config, _defaultProjectPath);
        return FormatOperationResult("Model Training", result, new
        {
            dataPath = dataPath ?? "project default",
            timeSeconds,
            metric,
            testSplit,
            experimentName
        });
    }

    /// <summary>
    /// Evaluate a trained model.
    /// </summary>
    [Description("Evaluate a trained model on test data")]
    public async Task<string> EvaluateModelAsync(
        [Description("Experiment ID to evaluate (optional, uses latest if not specified)")] string? experimentId = null,
        [Description("Path to test data CSV file (optional)")] string? testDataPath = null)
    {
        var result = await _projectManager.EvaluateModelAsync(experimentId, testDataPath, _defaultProjectPath);
        return FormatOperationResult("Model Evaluation", result, new
        {
            experimentId = experimentId ?? "latest",
            testDataPath = testDataPath ?? "default test set"
        });
    }

    /// <summary>
    /// Make predictions using a trained model.
    /// </summary>
    [Description("Make predictions using a trained model")]
    public async Task<string> PredictAsync(
        [Description("Path to input data CSV file for predictions")] string inputDataPath,
        [Description("Path where predictions will be saved")] string outputPath,
        [Description("Experiment ID to use for predictions (optional, uses production model if not specified)")] string? experimentId = null)
    {
        var result = await _projectManager.PredictAsync(inputDataPath, outputPath, experimentId, _defaultProjectPath);
        return FormatOperationResult("Prediction", result, new
        {
            inputDataPath,
            outputPath,
            experimentId = experimentId ?? "production"
        });
    }

    /// <summary>
    /// List all experiments in the project.
    /// </summary>
    [Description("List all experiments with their metrics")]
    public async Task<string> ListExperimentsAsync()
    {
        var experiments = await _projectManager.ListExperimentsAsync(_defaultProjectPath);

        if (experiments.Count == 0)
        {
            return JsonSerializer.Serialize(new
            {
                success = true,
                message = "No experiments found in this project.",
                experiments = Array.Empty<object>()
            }, new JsonSerializerOptions { WriteIndented = true });
        }

        var formattedExperiments = experiments
            .OrderByDescending(e => e.Timestamp)
            .Select(e => new
            {
                id = e.Id,
                name = e.Name,
                trainer = e.Trainer,
                metric = e.MetricName,
                value = e.MetricValue,
                timestamp = e.Timestamp.ToString("yyyy-MM-dd HH:mm:ss"),
                isProduction = e.IsProduction
            })
            .ToList();

        var best = experiments.OrderByDescending(e => e.MetricValue).FirstOrDefault();

        return JsonSerializer.Serialize(new
        {
            success = true,
            totalExperiments = experiments.Count,
            bestExperiment = best != null ? new
            {
                id = best.Id,
                name = best.Name,
                trainer = best.Trainer,
                metricValue = best.MetricValue
            } : null,
            experiments = formattedExperiments
        }, new JsonSerializerOptions { WriteIndented = true });
    }

    /// <summary>
    /// Promote an experiment to production.
    /// </summary>
    [Description("Promote an experiment to production status")]
    public async Task<string> PromoteExperimentAsync(
        [Description("Experiment ID to promote to production")] string experimentId)
    {
        var result = await _projectManager.PromoteExperimentAsync(experimentId, _defaultProjectPath);
        return FormatOperationResult("Experiment Promotion", result, new
        {
            experimentId,
            promotedTo = "production"
        });
    }

    /// <summary>
    /// Get dataset information.
    /// </summary>
    [Description("Get detailed information about a dataset")]
    public async Task<string> GetDatasetInfoAsync(
        [Description("Path to the dataset CSV file")] string dataPath)
    {
        var result = await _projectManager.GetDatasetInfoAsync(dataPath, _defaultProjectPath);
        return FormatOperationResult("Dataset Info", result, new
        {
            dataPath
        });
    }

    /// <summary>
    /// Execute preprocessing scripts.
    /// </summary>
    [Description("Execute preprocessing scripts on project data")]
    public async Task<string> PreprocessDataAsync()
    {
        var projectPath = _defaultProjectPath ?? Directory.GetCurrentDirectory();
        var result = await _projectManager.PreprocessDataAsync(projectPath);
        return FormatOperationResult("Preprocessing", result, new
        {
            projectPath
        });
    }

    #endregion

    #region Result Formatting

    private static string FormatOperationResult(string operation, MLoopOperationResult result, object parameters)
    {
        var response = new
        {
            operation,
            success = result.Success,
            exitCode = result.ExitCode,
            parameters,
            output = result.Output,
            error = result.Error,
            data = result.Data
        };

        return JsonSerializer.Serialize(response, new JsonSerializerOptions
        {
            WriteIndented = true,
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
        });
    }

    #endregion
}
