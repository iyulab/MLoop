using System.CommandLine;
using System.Reflection;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.AutoML;
using MLoop.Core.Evaluation;
using MLoop.Core.Storage;
using MLoop.Core.Data;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop init - Initializes a new MLoop project with multi-model support
/// </summary>
public static class InitCommand
{
    public static Command Create()
    {
        var projectNameArg = new Argument<string>("project-name")
        {
            Description = "Name of the project to create (use '.' for current directory)"
        };

        var taskOption = new Option<string>("--task", "-t")
        {
            Description = "ML task type (e.g., regression, binary-classification, clustering, forecasting, recommendation)",
            DefaultValueFactory = _ => "binary-classification"
        };

        var nameOption = new Option<string>("--name", "-n")
        {
            Description = $"Initial model name (default: '{ConfigDefaults.DefaultModelName}')",
            DefaultValueFactory = _ => ConfigDefaults.DefaultModelName
        };

        var labelOption = new Option<string>("--label", "-l")
        {
            Description = "Label column name (can be set later in mloop.yaml)",
            DefaultValueFactory = _ => "Label"
        };

        var forceOption = new Option<bool>("--force", "-f")
        {
            Description = "Reinitialize existing project (preserves .mloop/scripts/ directory)",
            DefaultValueFactory = _ => false
        };

        var command = new Command("init", "Initialize a new ML project with multi-model support");
        command.Arguments.Add(projectNameArg);
        command.Options.Add(taskOption);
        command.Options.Add(nameOption);
        command.Options.Add(labelOption);
        command.Options.Add(forceOption);

        command.SetAction((parseResult) =>
        {
            var projectName = parseResult.GetValue(projectNameArg)!;
            var task = parseResult.GetValue(taskOption)!;
            var model = parseResult.GetValue(nameOption)!;
            var label = parseResult.GetValue(labelOption)!;
            var force = parseResult.GetValue(forceOption);
            return ExecuteAsync(projectName, task, model, label, force);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string projectName,
        string task,
        string modelName,
        string labelColumn,
        bool force)
    {
        try
        {
            AnsiConsole.MarkupLine($"[blue]Initializing MLoop project:[/] [green]{projectName}[/]");
            AnsiConsole.WriteLine();

            // Validate project name
            if (string.IsNullOrWhiteSpace(projectName))
            {
                AnsiConsole.MarkupLine("[red]Error:[/] Project name cannot be empty");
                return 1;
            }

            if (projectName != "." && projectName.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] Project name contains invalid characters");
                return 1;
            }

            // Validate task
            var validTasks = new[]
            {
                "binary-classification", "multiclass-classification", "regression",
                "anomaly-detection", "clustering", "ranking", "forecasting",
                "time-series-anomaly", "recommendation", "image-classification",
                "object-detection", "text-classification", "sentence-similarity",
                "ner", "question-answering"
            };
            if (!validTasks.Contains(task))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Invalid task type. Valid options: {string.Join(", ", validTasks)}");
                return 1;
            }

            // Validate model name
            if (!IsValidModelName(modelName))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Invalid model name '{modelName}'. " +
                    "Model names must be lowercase alphanumeric with hyphens, 2-50 characters.");
                return 1;
            }

            // For unsupervised tasks, clear default label if user didn't explicitly set one
            // (AutoMLRunner.RequiresLabel — single source shared with merge/train/validate).
            if (!AutoMLRunner.RequiresLabel(task) && labelColumn == "Label")
                labelColumn = "";

            // Resolve project path (allow "." for current directory)
            var projectPath = projectName == "."
                ? Directory.GetCurrentDirectory()
                : Path.Combine(Directory.GetCurrentDirectory(), projectName);

            var mloopDir = Path.Combine(projectPath, ".mloop");
            var isReinitialize = Directory.Exists(mloopDir);

            // Check if directory/project already exists
            if (projectName != "." && Directory.Exists(projectPath) && !force)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Directory '{projectName}' already exists");
                AnsiConsole.MarkupLine("[yellow]Tip:[/] Use --force to reinitialize (preserves scripts/)");
                return 1;
            }

            if (isReinitialize && !force)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Project already initialized at {projectPath}");
                AnsiConsole.MarkupLine("[yellow]Tip:[/] Use --force to reinitialize (preserves scripts/)");
                return 1;
            }

            // Handle reinitialization with script preservation
            string? scriptsBackupPath = null;
            if (isReinitialize && force)
            {
                AnsiConsole.MarkupLine("[yellow]Reinitializing existing project[/]");

                var scriptsDir = Path.Combine(mloopDir, "scripts");
                if (Directory.Exists(scriptsDir))
                {
                    scriptsBackupPath = Path.Combine(Path.GetTempPath(), $"mloop_scripts_backup_{Guid.NewGuid()}");
                    Directory.CreateDirectory(scriptsBackupPath);
                    CopyDirectory(scriptsDir, scriptsBackupPath);
                    AnsiConsole.MarkupLine($"[green]>[/] Backed up scripts/ to temp location");
                }

                Directory.Delete(mloopDir, recursive: true);
                AnsiConsole.MarkupLine($"[green]>[/] Cleaned .mloop directory");
                AnsiConsole.WriteLine();
            }

            var displayName = projectName == "." ? Path.GetFileName(projectPath) : projectName;

            await AnsiConsole.Progress()
                .StartAsync(async ctx =>
                {
                    var progressTask = ctx.AddTask("[green]Setting up project...[/]");

                    var fileSystem = new FileSystemManager();

                    // Step 1: Create directory structure
                    progressTask.Description = "[green]Creating directory structure...[/]";
                    await CreateDirectoryStructure(fileSystem, projectPath, modelName, task);
                    progressTask.Increment(30);

                    // Step 2: Create configuration files
                    progressTask.Description = "[green]Creating configuration files...[/]";
                    await CreateConfigurationFiles(fileSystem, projectPath, displayName, task, modelName, labelColumn);
                    progressTask.Increment(30);

                    // Step 3: Create documentation
                    progressTask.Description = "[green]Creating documentation...[/]";
                    await CreateDocumentation(fileSystem, projectPath, displayName, task, modelName);
                    progressTask.Increment(20);

                    // Step 4: Create .gitignore
                    progressTask.Description = "[green]Creating .gitignore...[/]";
                    await CreateGitIgnore(fileSystem, projectPath);
                    progressTask.Increment(10);

                    // Step 5: Create example extensibility scripts
                    progressTask.Description = "[green]Creating example scripts...[/]";
                    await CreateExampleScripts(fileSystem, projectPath);
                    progressTask.Increment(10);

                    progressTask.Description = "[green]Project initialized![/]";

                    // Restore scripts if this was a reinitialize
                    if (scriptsBackupPath != null && Directory.Exists(scriptsBackupPath))
                    {
                        progressTask.Description = "[green]Restoring scripts...[/]";
                        var scriptsDir = Path.Combine(mloopDir, "scripts");
                        CopyDirectory(scriptsBackupPath, scriptsDir);
                        Directory.Delete(scriptsBackupPath, recursive: true);
                        AnsiConsole.MarkupLine($"[green]>[/] Restored scripts/ directory");
                    }
                });

            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[green]>[/] Project '{displayName}' created successfully!");
            AnsiConsole.WriteLine();

            // Display configuration summary
            var table = new Table()
                .BorderColor(Color.Grey)
                .AddColumn("Setting")
                .AddColumn("Value");

            table.AddRow("Project", displayName);
            table.AddRow("Model", $"[cyan]{modelName}[/]");
            table.AddRow("Task", task);
            table.AddRow("Label Column", labelColumn);

            AnsiConsole.Write(table);
            AnsiConsole.WriteLine();

            var isDirectoryBased = DataLoaderFactory.IsDirectoryBased(task);

            AnsiConsole.MarkupLine("[yellow]Next steps:[/]");
            if (projectName != ".")
            {
                AnsiConsole.MarkupLine($"  1. cd {projectName}");
            }
            var isObjectDetection = task.Equals("object-detection", StringComparison.OrdinalIgnoreCase);
            if (isObjectDetection)
            {
                AnsiConsole.MarkupLine("  2. Edit datasets/coco/annotations.json  [cyan]# COCO format; add images + bounding boxes[/]");
                AnsiConsole.MarkupLine("  3. mloop runtime install torch  [cyan]# one-time TorchSharp runtime[/]");
                AnsiConsole.MarkupLine("  4. mloop train  [cyan]# Auto-detects datasets/coco/[/]");
            }
            else if (isDirectoryBased)
            {
                AnsiConsole.MarkupLine("  2. Place images under datasets/images/<class>/  [cyan]# folder name = label[/]");
                AnsiConsole.MarkupLine("  3. mloop runtime install tf  [cyan]# one-time TensorFlow CPU runtime (~182MB)[/]");
                AnsiConsole.MarkupLine("  4. mloop train  [cyan]# Auto-detects datasets/images/[/]");
            }
            else
            {
                AnsiConsole.MarkupLine("  2. Place your training data in datasets/train.csv");
                AnsiConsole.MarkupLine("  3. Edit mloop.yaml to set your label column");
                AnsiConsole.MarkupLine("  4. mloop train  [cyan]# Auto-detects datasets/train.csv[/]");
            }
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine("[grey]Folder structure:[/]");
            if (isObjectDetection)
                AnsiConsole.MarkupLine("  datasets/coco/              [cyan]# COCO annotations.json + images[/]");
            else if (isDirectoryBased)
                AnsiConsole.MarkupLine("  datasets/images/<class>/    [cyan]# Training images (folder = label)[/]");
            else
                AnsiConsole.MarkupLine("  datasets/                   [cyan]# Training data (train.csv)[/]");
            AnsiConsole.MarkupLine($"  models/{modelName}/staging/    [cyan]# Experimental models[/]");
            AnsiConsole.MarkupLine($"  models/{modelName}/production/ [cyan]# Promoted production model[/]");
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine("[grey]Multi-model support:[/]");
            AnsiConsole.MarkupLine("  mloop train --name churn     [cyan]# Train different model[/]");
            AnsiConsole.MarkupLine("  mloop list --name churn      [cyan]# List model experiments[/]");
            AnsiConsole.MarkupLine("  mloop predict --name churn   [cyan]# Predict with specific model[/]");
            AnsiConsole.WriteLine();

            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "init");
            return 1;
        }
    }

    private static async Task CreateDirectoryStructure(
        IFileSystemManager fileSystem,
        string projectPath,
        string modelName,
        string task)
    {
        // Create main project directory
        await fileSystem.CreateDirectoryAsync(projectPath);

        // Create .mloop directory (internal)
        var mloopPath = fileSystem.CombinePath(projectPath, ".mloop");
        await fileSystem.CreateDirectoryAsync(mloopPath);

        // MLOps convention: datasets/ folder for training data
        var datasetsPath = fileSystem.CombinePath(projectPath, "datasets");
        await fileSystem.CreateDirectoryAsync(datasetsPath);

        // Directory-based tasks read a directory rather than a CSV. Image classification
        // uses datasets/images/<class>/; object detection uses a COCO annotations file under
        // datasets/coco/. Scaffold the appropriate layout plus a README documenting it.
        if (task.Equals("object-detection", StringComparison.OrdinalIgnoreCase))
        {
            var cocoPath = fileSystem.CombinePath(datasetsPath, "coco");
            await fileSystem.CreateDirectoryAsync(cocoPath);
            await fileSystem.WriteTextAsync(
                fileSystem.CombinePath(cocoPath, "annotations.json"),
                "{\n" +
                "  \"images\": [\n" +
                "    { \"id\": 1, \"file_name\": \"img001.jpg\", \"width\": 640, \"height\": 480 }\n" +
                "  ],\n" +
                "  \"annotations\": [\n" +
                "    { \"image_id\": 1, \"category_id\": 1, \"bbox\": [100, 120, 80, 60] }\n" +
                "  ],\n" +
                "  \"categories\": [\n" +
                "    { \"id\": 1, \"name\": \"defect\" }\n" +
                "  ]\n" +
                "}\n");
            await fileSystem.WriteTextAsync(
                fileSystem.CombinePath(cocoPath, "README.txt"),
                "Object detection dataset layout (COCO format):\n\n" +
                "  datasets/coco/\n" +
                "  ├── annotations.json   # COCO: images[], annotations[] (bbox=[x,y,w,h]), categories[]\n" +
                "  ├── img001.jpg\n" +
                "  └── img002.jpg ...\n\n" +
                "Each annotation's bbox is [x, y, width, height] in absolute pixels; MLoop converts it\n" +
                "to the x0 y0 x1 y1 order the trainer expects. file_name is resolved relative to this folder.\n" +
                "Edit annotations.json (the scaffold is a template), then run:\n" +
                "  mloop runtime install torch    # one-time TorchSharp runtime\n" +
                "  mloop train --task object-detection\n\n" +
                "Supported conventional file names: annotations.json, _annotations.coco.json (Roboflow).\n");
        }
        else if (DataLoaderFactory.IsDirectoryBased(task))
        {
            var imagesPath = fileSystem.CombinePath(datasetsPath, "images");
            await fileSystem.CreateDirectoryAsync(imagesPath);
            await fileSystem.WriteTextAsync(
                fileSystem.CombinePath(imagesPath, "README.txt"),
                "Image classification dataset layout (folder name = class label):\n\n" +
                "  datasets/images/\n" +
                "  ├── OK/   img001.jpg img002.jpg ...\n" +
                "  └── NG/   img101.jpg img102.jpg ...\n\n" +
                "Place each class's images in its own subfolder, then run:\n" +
                "  mloop runtime install tf      # one-time TensorFlow CPU runtime (~182MB)\n" +
                "  mloop train --task image-classification\n\n" +
                "Supported extensions: .jpg .jpeg .png .bmp .gif\n");
        }

        // MLOps convention: models/{modelName}/ folder structure
        var modelPath = fileSystem.CombinePath(projectPath, ExperimentLayout.ModelsDirectory, modelName);
        await fileSystem.CreateDirectoryAsync(fileSystem.CombinePath(modelPath, ExperimentLayout.StagingDirectory));
        await fileSystem.CreateDirectoryAsync(fileSystem.CombinePath(modelPath, ExperimentLayout.ProductionDirectory));

        // Extensibility: scripts/ folder structure for hooks and metrics
        // Initialize hook directories (pre-train, post-train, pre-predict, post-evaluate)
        var hookEngine = new MLoop.Core.Hooks.HookEngine(projectPath, new NullLogger());
        hookEngine.InitializeDirectories();

        await fileSystem.CreateDirectoryAsync(fileSystem.CombinePath(mloopPath, "scripts", "metrics"));
    }

    private static async Task CreateConfigurationFiles(
        IFileSystemManager fileSystem,
        string projectPath,
        string projectName,
        string task,
        string modelName,
        string labelColumn)
    {
        // Create .mloop/config.json (project metadata)
        var configData = new
        {
            project = projectName,
            version = "1.0",
            created_at = DateTime.UtcNow,
            mloop_version = typeof(InitCommand).Assembly
                .GetCustomAttribute<System.Reflection.AssemblyInformationalVersionAttribute>()
                ?.InformationalVersion ?? typeof(InitCommand).Assembly.GetName().Version?.ToString() ?? "unknown"
        };

        var configPath = fileSystem.CombinePath(projectPath, ".mloop", "config.json");
        await fileSystem.WriteJsonAsync(configPath, configData);

        // Create mloop.yaml with new multi-model schema
        var yamlContent = GetYamlTemplate(projectName, task, modelName, labelColumn);
        var yamlPath = fileSystem.CombinePath(projectPath, "mloop.yaml");
        await fileSystem.WriteTextAsync(yamlPath, yamlContent);
    }

    private static async Task CreateDocumentation(
        IFileSystemManager fileSystem,
        string projectPath,
        string projectName,
        string task,
        string modelName)
    {
        var readme = $@"# {projectName}

MLoop machine learning project for {task}.

## Quick Start

### 1. Prepare Your Data

Place your training data in `datasets/train.csv`:

```bash
# Your CSV should have:
# - One column for the label (target variable)
# - Other columns as features
```

### 2. Train a Model

```bash
# Train the default model
mloop train --label <your-label-column>

# Train with specific model name
mloop train --name {modelName} --label <your-label-column>
```

### 3. Make Predictions

```bash
# Predict using default model
mloop predict new-data.csv

# Predict using specific model
mloop predict new-data.csv --name {modelName}
```

## Multi-Model Support

MLoop supports multiple models per project. Each model has its own:
- Experiments (staging)
- Production deployment
- Metrics history

```bash
# Train different models for different targets
mloop train --name churn --label Churned --task binary-classification
mloop train --name revenue --label Revenue --task regression

# List experiments per model
mloop list --name churn
mloop list --name revenue

# Promote and predict
mloop promote exp-001 --name churn
mloop predict new-data.csv --name churn
```

## Project Structure

```
{projectName}/
├── .mloop/              # Internal MLoop metadata (gitignored)
│   └── scripts/         # Hooks and custom metrics
├── mloop.yaml           # User configuration
├── datasets/            # Training data (train.csv)
└── models/
    └── {modelName}/
        ├── staging/     # Experimental models (exp-001, exp-002, ...)
        └── production/  # Promoted production model
```

## Configuration

Edit `mloop.yaml` to configure:
- Model definitions (task, label, training settings)
- Default data paths

## Commands

- `mloop init` - Initialize a new project
- `mloop train` - Train a model
- `mloop predict` - Make predictions
- `mloop list` - List experiments
- `mloop promote` - Promote experiment to production
- `mloop evaluate` - Evaluate model performance

## Documentation

For more information, see [MLoop documentation](https://github.com/iyulab/MLoop).
";

        var readmePath = fileSystem.CombinePath(projectPath, "README.md");
        await fileSystem.WriteTextAsync(readmePath, readme);
    }

    private static async Task CreateGitIgnore(IFileSystemManager fileSystem, string projectPath)
    {
        var gitignore = @"# MLoop user project .gitignore

# .NET Build outputs (if any scripts)
bin/
obj/
*.user
*.suo

# MLoop internal (keep metadata, ignore binaries)
.mloop/cache/
.mloop/.cache/

# Model binaries (large files)
models/*/staging/*/model.zip
models/*/production/model.zip
models/*/staging/*/training.log

# Prediction outputs
predictions/

# OS
.DS_Store
Thumbs.db
*.swp
";

        var gitignorePath = fileSystem.CombinePath(projectPath, ".gitignore");
        await fileSystem.WriteTextAsync(gitignorePath, gitignore);
    }

    private static async Task CreateExampleScripts(IFileSystemManager fileSystem, string projectPath)
    {
        // Create example hook README
        var hookReadme = @"# MLoop Hooks

Place your custom hook scripts in this directory. Hooks execute at lifecycle points during training.

## Example Hook

```csharp
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Extensibility;

public class DataValidationHook : IMLoopHook
{
    public string Name => ""Data Validation"";

    public async Task<HookResult> ExecuteAsync(HookContext ctx)
    {
        var preview = ctx.DataView.Preview(maxRows: 100);
        var rowCount = preview.RowView.Length;

        if (rowCount < 100)
        {
            return HookResult.Abort($""Insufficient data: {rowCount} rows, need at least 100"");
        }

        ctx.Logger.Info($""Data validation passed: {rowCount} rows"");
        return HookResult.Continue();
    }
}
```

## Usage

1. Create a new .cs file in this directory
2. Implement the `IMLoopHook` interface
3. Run `mloop train` - hooks are auto-discovered and executed

See: docs/EXTENSIBILITY.md for more information
";

        var hookReadmePath = fileSystem.CombinePath(projectPath, ".mloop", "scripts", "hooks", "README.md");
        await fileSystem.WriteTextAsync(hookReadmePath, hookReadme);

        // Create example metric README
        var metricReadme = @"# MLoop Custom Metrics

Place your custom metric scripts in this directory. Metrics evaluate model performance using business-specific logic.

## Example Metric

```csharp
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Extensibility;

public class ProfitMetric : IMLoopMetric
{
    public string Name => ""Expected Profit"";
    public bool HigherIsBetter => true;

    private const double PROFIT_PER_TP = 100.0;
    private const double LOSS_PER_FP = -50.0;

    public async Task<double> CalculateAsync(MetricContext ctx)
    {
        var metrics = ctx.MLContext.BinaryClassification.Evaluate(ctx.Predictions);

        var profit = (metrics.PositiveRecall * PROFIT_PER_TP) +
                     ((1 - metrics.NegativeRecall) * LOSS_PER_FP);

        ctx.Logger.Info($""Calculated profit: ${profit:F2}"");
        return await Task.FromResult(profit);
    }
}
```

## Usage

1. Create a new .cs file in this directory
2. Implement the `IMLoopMetric` interface
3. Run `mloop train` - metrics are auto-discovered and calculated

See: docs/EXTENSIBILITY.md for more information
";

        var metricReadmePath = fileSystem.CombinePath(projectPath, ".mloop", "scripts", "metrics", "README.md");
        await fileSystem.WriteTextAsync(metricReadmePath, metricReadme);
    }

    internal static string GetYamlTemplate(string projectName, string task, string modelName, string labelColumn)
    {
        // Task→primary-metric mapping lives in the shared TaskMetadata source of truth so the
        // yaml example, the promotion gate, and AutoML all agree (TD-06). Tasks without a
        // canonical metric (object detection, unknown) defer to "auto".
        var metricExample = TaskMetadata.PrimaryMetricOrAuto(task);

        var taskSpecificFields = task switch
        {
            "forecasting" => "\n    horizon: 10",
            "ranking" => "\n    group_column: query_id",
            "recommendation" => "\n    user_column: user_id\n    item_column: item_id",
            "clustering" => "\n    # num_clusters: auto",
            _ => ""
        };

        return $@"# MLoop Project Configuration
# Multi-model MLOps with minimal cost

project: {projectName}

# Model definitions
# Each model has its own experiment namespace and production slot
models:
  {modelName}:
    task: {task}
    label: {labelColumn}{taskSpecificFields}
    description: Default model for {task}
    training:
      # time_limit_seconds: 300   # optional — omit to let mloop auto-estimate the training
      #                             time from data size ([30, 1800]s). Set a fixed value to override.
      metric: {metricExample}
      test_split: 0.2

# Add more models as needed:
# models:
#   churn:
#     task: binary-classification
#     label: Churned
#   revenue:
#     task: regression
#     label: Revenue

# Default data paths
data:
  train: datasets/train.csv
  test: datasets/test.csv
";
    }

    internal static bool IsValidModelName(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            return false;

        if (name.Length < 2 || name.Length > 50)
            return false;

        // Reserved names
        var reserved = new[] { "staging", "production", "temp", "cache", "index", "registry" };
        if (reserved.Contains(name, StringComparer.OrdinalIgnoreCase))
            return false;

        // Must be lowercase alphanumeric with hyphens
        return System.Text.RegularExpressions.Regex.IsMatch(name, @"^[a-z][a-z0-9]*(-[a-z0-9]+)*$");
    }

    private static void CopyDirectory(string sourceDir, string destDir)
        => MLoop.Ops.Services.FilePromotionManager.CopyDirectory(sourceDir, destDir);

    private class NullLogger : MLoop.Extensibility.Preprocessing.ILogger
    {
        public void Debug(string message) { }
        public void Info(string message) { }
        public void Warning(string message) { }
        public void Error(string message) { }
        public void Error(string message, Exception exception) { }
    }
}
