using MLoop.AIAgent.Core.Models;
using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;

namespace MLoop.AIAgent.Core;

/// <summary>
/// Manages MLoop project lifecycle through CLI wrapper
/// </summary>
public class MLoopProjectManager
{
    private readonly string _mloopCliPath;
    private readonly string? _dotnetPath;

    /// <summary>
    /// Initialize MLoopProjectManager
    /// </summary>
    /// <param name="mloopCliPath">Path to MLoop.CLI.csproj (optional, defaults to standard location)</param>
    /// <param name="dotnetPath">Path to dotnet executable (optional, uses PATH if not specified)</param>
    public MLoopProjectManager(string? mloopCliPath = null, string? dotnetPath = null)
    {
        _mloopCliPath = mloopCliPath ?? FindMLoopCli();
        _dotnetPath = dotnetPath ?? "dotnet";
    }

    /// <summary>
    /// Initialize a new MLoop project
    /// </summary>
    public async Task<MLoopOperationResult> InitializeProjectAsync(MLoopProjectConfig config)
    {
        var args = new List<string>
        {
            "init",
            config.ProjectName,
            "--data", config.DataPath,
            "--label", config.LabelColumn,
            "--task", config.TaskType
        };

        if (!string.IsNullOrEmpty(config.ProjectDirectory))
        {
            args.AddRange(new[] { "--path", config.ProjectDirectory });
        }

        var result = await ExecuteCliCommandAsync(args.ToArray());

        // Extract project directory from output
        if (result.Success)
        {
            var projectDirMatch = Regex.Match(result.Output, @"Project created at: (.+)");
            if (projectDirMatch.Success)
            {
                result.Data["ProjectDirectory"] = projectDirMatch.Groups[1].Value.Trim();
            }
        }

        return result;
    }

    /// <summary>
    /// Execute preprocessing scripts
    /// </summary>
    public async Task<MLoopOperationResult> PreprocessDataAsync(string projectPath)
    {
        var args = new[] { "preprocess" };
        return await ExecuteCliCommandAsync(args, projectPath);
    }

    /// <summary>
    /// Train a model using AutoML
    /// </summary>
    public async Task<MLoopOperationResult> TrainModelAsync(
        MLoopTrainingConfig config,
        string? projectPath = null)
    {
        var args = new List<string>
        {
            "train",
            "--time", config.TimeSeconds.ToString(),
            "--metric", config.Metric,
            "--test-split", config.TestSplit.ToString("F2")
        };

        if (!string.IsNullOrEmpty(config.DataPath))
        {
            args.AddRange(new[] { "--data", config.DataPath });
        }

        if (!string.IsNullOrEmpty(config.ExperimentName))
        {
            args.AddRange(new[] { "--name", config.ExperimentName });
        }

        var result = await ExecuteCliCommandAsync(args.ToArray(), projectPath);

        // Extract experiment info from output
        if (result.Success)
        {
            ExtractTrainingMetrics(result);
        }

        return result;
    }

    /// <summary>
    /// Evaluate model performance
    /// </summary>
    public async Task<MLoopOperationResult> EvaluateModelAsync(
        string? experimentId = null,
        string? testDataPath = null,
        string? projectPath = null)
    {
        var args = new List<string> { "evaluate" };

        if (!string.IsNullOrEmpty(experimentId))
        {
            args.AddRange(new[] { "--experiment", experimentId });
        }

        if (!string.IsNullOrEmpty(testDataPath))
        {
            args.AddRange(new[] { "--test-data", testDataPath });
        }

        var result = await ExecuteCliCommandAsync(args.ToArray(), projectPath);

        // Extract evaluation metrics
        if (result.Success)
        {
            ExtractEvaluationMetrics(result);
        }

        return result;
    }

    /// <summary>
    /// Make predictions using trained model
    /// </summary>
    public async Task<MLoopOperationResult> PredictAsync(
        string inputDataPath,
        string outputPath,
        string? experimentId = null,
        string? projectPath = null)
    {
        var args = new List<string>
        {
            "predict",
            "--input", inputDataPath,
            "--output", outputPath
        };

        if (!string.IsNullOrEmpty(experimentId))
        {
            args.AddRange(new[] { "--experiment", experimentId });
        }

        return await ExecuteCliCommandAsync(args.ToArray(), projectPath);
    }

    /// <summary>
    /// List all experiments in the project
    /// </summary>
    public async Task<List<MLoopExperiment>> ListExperimentsAsync(string? projectPath = null)
    {
        var args = new[] { "list", "--format", "json" };
        var result = await ExecuteCliCommandAsync(args, projectPath);

        var experiments = new List<MLoopExperiment>();

        if (result.Success)
        {
            // Parse JSON output to extract experiments
            // For now, parse text output with regex
            var experimentMatches = Regex.Matches(
                result.Output,
                @"(?<id>[a-f0-9-]+)\s+(?<name>\S+)\s+(?<trainer>\w+)\s+(?<metric>[\d.]+)\s+(?<date>[\d-]+\s+[\d:]+)");

            foreach (Match match in experimentMatches)
            {
                experiments.Add(new MLoopExperiment
                {
                    Id = match.Groups["id"].Value,
                    Name = match.Groups["name"].Value,
                    Trainer = match.Groups["trainer"].Value,
                    MetricValue = double.Parse(match.Groups["metric"].Value),
                    MetricName = "Accuracy", // Default, should be extracted from output
                    Timestamp = DateTime.Parse(match.Groups["date"].Value),
                    IsProduction = false
                });
            }
        }

        return experiments;
    }

    /// <summary>
    /// Promote an experiment to production
    /// </summary>
    public async Task<MLoopOperationResult> PromoteExperimentAsync(
        string experimentId,
        string? projectPath = null)
    {
        var args = new[] { "promote", experimentId };
        return await ExecuteCliCommandAsync(args, projectPath);
    }

    /// <summary>
    /// Get dataset information
    /// </summary>
    public async Task<MLoopOperationResult> GetDatasetInfoAsync(
        string dataPath,
        string? projectPath = null)
    {
        var args = new[] { "info", "--data", dataPath };
        return await ExecuteCliCommandAsync(args, projectPath);
    }

    /// <summary>
    /// Execute a CLI command
    /// </summary>
    private async Task<MLoopOperationResult> ExecuteCliCommandAsync(
        string[] args,
        string? workingDirectory = null)
    {
        var processStartInfo = new ProcessStartInfo
        {
            FileName = _dotnetPath,
            Arguments = $"run --project \"{_mloopCliPath}\" -- {string.Join(" ", args)}",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WorkingDirectory = workingDirectory ?? Environment.CurrentDirectory
        };

        var outputBuilder = new StringBuilder();
        var errorBuilder = new StringBuilder();

        using var process = new Process { StartInfo = processStartInfo };

        process.OutputDataReceived += (sender, e) =>
        {
            if (e.Data != null)
            {
                outputBuilder.AppendLine(e.Data);
            }
        };

        process.ErrorDataReceived += (sender, e) =>
        {
            if (e.Data != null)
            {
                errorBuilder.AppendLine(e.Data);
            }
        };

        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        await process.WaitForExitAsync();

        var output = outputBuilder.ToString();
        var error = errorBuilder.ToString();
        var exitCode = process.ExitCode;

        return new MLoopOperationResult
        {
            Success = exitCode == 0,
            ExitCode = exitCode,
            Output = output,
            Error = string.IsNullOrWhiteSpace(error) ? null : error
        };
    }

    /// <summary>
    /// Extract training metrics from CLI output
    /// </summary>
    private void ExtractTrainingMetrics(MLoopOperationResult result)
    {
        // Extract experiment ID
        var expIdMatch = Regex.Match(result.Output, @"Experiment ID: ([a-f0-9-]+)");
        if (expIdMatch.Success)
        {
            result.Data["ExperimentId"] = expIdMatch.Groups[1].Value;
        }

        // Extract best trainer
        var trainerMatch = Regex.Match(result.Output, @"Best trainer: (\w+)");
        if (trainerMatch.Success)
        {
            result.Data["BestTrainer"] = trainerMatch.Groups[1].Value;
        }

        // Extract metric value
        var metricMatch = Regex.Match(result.Output, @"(?:Accuracy|F1Score|RSquared|RMSE|MAE):\s*([\d.]+)");
        if (metricMatch.Success)
        {
            result.Data["MetricValue"] = double.Parse(metricMatch.Groups[1].Value);
        }
    }

    /// <summary>
    /// Extract evaluation metrics from CLI output
    /// </summary>
    private void ExtractEvaluationMetrics(MLoopOperationResult result)
    {
        // Extract all metric values
        var metricMatches = Regex.Matches(
            result.Output,
            @"(\w+):\s*([\d.]+)");

        foreach (Match match in metricMatches)
        {
            var metricName = match.Groups[1].Value;
            var metricValue = double.Parse(match.Groups[2].Value);
            result.Data[metricName] = metricValue;
        }
    }

    /// <summary>
    /// Find MLoop CLI project path
    /// </summary>
    private string FindMLoopCli()
    {
        // Try common locations
        var possiblePaths = new[]
        {
            Path.Combine(Environment.CurrentDirectory, "src", "MLoop.CLI", "MLoop.CLI.csproj"),
            Path.Combine(Environment.CurrentDirectory, "..", "MLoop.CLI", "MLoop.CLI.csproj"),
            Path.Combine(Environment.CurrentDirectory, "..", "..", "src", "MLoop.CLI", "MLoop.CLI.csproj")
        };

        foreach (var path in possiblePaths)
        {
            if (File.Exists(path))
            {
                return Path.GetFullPath(path);
            }
        }

        // Default path
        return Path.Combine(Environment.CurrentDirectory, "src", "MLoop.CLI", "MLoop.CLI.csproj");
    }
}
