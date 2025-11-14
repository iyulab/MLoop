using System.CommandLine;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop init - Initializes a new MLoop project
/// </summary>
public static class InitCommand
{
    public static Command Create()
    {
        var projectNameArg = new Argument<string>("project-name")
        {
            Description = "Name of the project to create"
        };

        var taskOption = new Option<string>("--task", "-t")
        {
            Description = "ML task type: binary-classification, multiclass-classification, regression",
            DefaultValueFactory = _ => "binary-classification"
        };

        var forceOption = new Option<bool>("--force", "-f")
        {
            Description = "Reinitialize existing project (preserves .mloop/scripts/ directory)",
            DefaultValueFactory = _ => false
        };

        var command = new Command("init", "Initialize a new ML project");
        command.Arguments.Add(projectNameArg);
        command.Options.Add(taskOption);
        command.Options.Add(forceOption);

        command.SetAction((parseResult) =>
        {
            var projectName = parseResult.GetValue(projectNameArg)!;
            var task = parseResult.GetValue(taskOption)!;
            var force = parseResult.GetValue(forceOption);
            return ExecuteAsync(projectName, task, force);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(string projectName, string task, bool force)
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
            var validTasks = new[] { "binary-classification", "multiclass-classification", "regression" };
            if (!validTasks.Contains(task))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Invalid task type. Valid options: {string.Join(", ", validTasks)}");
                return 1;
            }

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
                AnsiConsole.MarkupLine("[yellow]⚠️  Reinitializing existing project[/]");

                var scriptsDir = Path.Combine(mloopDir, "scripts");
                if (Directory.Exists(scriptsDir))
                {
                    // Backup scripts to temp directory
                    scriptsBackupPath = Path.Combine(Path.GetTempPath(), $"mloop_scripts_backup_{Guid.NewGuid()}");
                    Directory.CreateDirectory(scriptsBackupPath);
                    CopyDirectory(scriptsDir, scriptsBackupPath);
                    AnsiConsole.MarkupLine($"[green]✓[/] Backed up scripts/ to temp location");
                }

                // Delete .mloop directory
                Directory.Delete(mloopDir, recursive: true);
                AnsiConsole.MarkupLine($"[green]✓[/] Cleaned .mloop directory");
                AnsiConsole.WriteLine();
            }

            await AnsiConsole.Progress()
                .StartAsync(async ctx =>
                {
                    var progressTask = ctx.AddTask("[green]Setting up project...[/]");

                    // Initialize filesystem and config
                    var fileSystem = new FileSystemManager();

                    // Step 1: Create directory structure
                    progressTask.Description = "[green]Creating directory structure...[/]";
                    await CreateDirectoryStructure(fileSystem, projectPath);
                    progressTask.Increment(30);

                    // Step 2: Create configuration files
                    progressTask.Description = "[green]Creating configuration files...[/]";
                    await CreateConfigurationFiles(fileSystem, projectPath, projectName, task);
                    progressTask.Increment(30);

                    // Step 3: Create documentation
                    progressTask.Description = "[green]Creating documentation...[/]";
                    await CreateDocumentation(fileSystem, projectPath, projectName, task);
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
                        AnsiConsole.MarkupLine($"[green]✓[/] Restored scripts/ directory");
                    }
                });

            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[green]✓[/] Project '{projectName}' created successfully!");
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine("[yellow]Next steps:[/]");
            AnsiConsole.MarkupLine($"  1. cd {projectName}");
            AnsiConsole.MarkupLine("  2. Place your training data in datasets/train.csv");
            AnsiConsole.MarkupLine("  3. Edit mloop.yaml to set your label column");
            AnsiConsole.MarkupLine("  4. mloop train  [cyan]# Auto-detects datasets/train.csv[/]");
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine("[grey]Folder structure:[/]");
            AnsiConsole.MarkupLine("  datasets/       [cyan]# Training data (train.csv, validation.csv, test.csv)[/]");
            AnsiConsole.MarkupLine("  models/staging/ [cyan]# Experimental models[/]");
            AnsiConsole.MarkupLine("  models/production/ [cyan]# Auto-promoted best model[/]");
            AnsiConsole.WriteLine();

            return 0;
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
            return 1;
        }
    }

    private static async Task CreateDirectoryStructure(IFileSystemManager fileSystem, string projectPath)
    {
        // Create main project directory
        await fileSystem.CreateDirectoryAsync(projectPath);

        // Create .mloop directory (internal)
        var mloopPath = fileSystem.CombinePath(projectPath, ".mloop");
        await fileSystem.CreateDirectoryAsync(mloopPath);

        // MLOps convention: datasets/ folder for training data
        await fileSystem.CreateDirectoryAsync(fileSystem.CombinePath(projectPath, "datasets"));

        // MLOps convention: models/ folder structure
        await fileSystem.CreateDirectoryAsync(fileSystem.CombinePath(projectPath, "models", "staging"));
        await fileSystem.CreateDirectoryAsync(fileSystem.CombinePath(projectPath, "models", "production"));

        // Extensibility: scripts/ folder structure for hooks and metrics
        await fileSystem.CreateDirectoryAsync(fileSystem.CombinePath(mloopPath, "scripts", "hooks"));
        await fileSystem.CreateDirectoryAsync(fileSystem.CombinePath(mloopPath, "scripts", "metrics"));
    }

    private static async Task CreateConfigurationFiles(
        IFileSystemManager fileSystem,
        string projectPath,
        string projectName,
        string task)
    {
        // Create .mloop/config.json
        var config = new MLoopConfig
        {
            ProjectName = projectName,
            Task = task,
            Training = new TrainingSettings
            {
                TimeLimitSeconds = 300,
                Metric = "accuracy",
                TestSplit = 0.2
            }
        };

        var configData = new
        {
            project_name = projectName,
            version = "0.1.0",
            task,
            created_at = DateTime.UtcNow,
            mloop_version = "0.1.0-alpha"
        };

        var configPath = fileSystem.CombinePath(projectPath, ".mloop", "config.json");
        await fileSystem.WriteJsonAsync(configPath, configData);

        // Initialize experiment index
        var experimentIndex = new
        {
            next_id = 1,
            experiments = Array.Empty<object>()
        };

        var indexPath = fileSystem.CombinePath(projectPath, ".mloop", "experiment-index.json");
        await fileSystem.WriteJsonAsync(indexPath, experimentIndex);

        // Initialize model registry
        var registry = new Dictionary<string, object>();
        var registryPath = fileSystem.CombinePath(projectPath, ".mloop", "registry.json");
        await fileSystem.WriteJsonAsync(registryPath, registry);

        // Create mloop.yaml from template
        var yamlContent = GetYamlTemplate(projectName, task);
        var yamlPath = fileSystem.CombinePath(projectPath, "mloop.yaml");
        await fileSystem.WriteTextAsync(yamlPath, yamlContent);
    }

    private static async Task CreateDocumentation(
        IFileSystemManager fileSystem,
        string projectPath,
        string projectName,
        string task)
    {
        var readme = $@"# {projectName}

MLoop machine learning project for {task}.

## Quick Start

### 1. Prepare Your Data

Place your training data in `data/processed/train.csv`:

```bash
# Your CSV should have:
# - One column for the label (target variable)
# - Other columns as features
```

### 2. Train a Model

```bash
mloop train data/processed/train.csv --label <your-label-column>
```

### 3. Make Predictions

```bash
mloop predict experiments/exp-001/model.zip data/new-data.csv
```

## Project Structure

```
{projectName}/
├── .mloop/              # Internal MLoop metadata (gitignored)
├── mloop.yaml           # User configuration
├── data/
│   ├── processed/       # Your training/test data
│   └── predictions/     # Prediction outputs
├── experiments/         # Training experiments
├── models/
│   ├── staging/         # Staging models
│   └── production/      # Production models
└── README.md
```

## Configuration

Edit `mloop.yaml` to customize:
- Training time limit
- Evaluation metric
- Test split ratio

## Commands

- `mloop train` - Train a model
- `mloop predict` - Make predictions
- `mloop evaluate` - Evaluate model performance
- `mloop experiment list` - List all experiments

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
experiments/*/model.zip
experiments/*/training.log
models/*/model.zip

# Prediction outputs
data/predictions/

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

        ctx.Logger.Info($""✅ Data validation passed: {rowCount} rows"");
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

    private static string GetYamlTemplate(string projectName, string task)
    {
        var labelColumnExample = task switch
        {
            "binary-classification" => "Label",
            "multiclass-classification" => "Category",
            "regression" => "Target",
            _ => "Label"
        };

        return $@"# MLoop Project Configuration
project_name: {projectName}
task: {task}
label_column: {labelColumnExample}

# Training settings (optional - defaults provided)
training:
  time_limit_seconds: 300
  metric: accuracy
  test_split: 0.2

# Data paths (optional)
data:
  train: data/processed/train.csv
  test: data/processed/test.csv

# Model output (optional)
model:
  output_dir: models/staging
";
    }

    /// <summary>
    /// Recursively copies a directory and all its contents.
    /// </summary>
    private static void CopyDirectory(string sourceDir, string destDir)
    {
        // Create destination directory
        Directory.CreateDirectory(destDir);

        // Copy files
        foreach (var file in Directory.GetFiles(sourceDir))
        {
            var fileName = Path.GetFileName(file);
            var destFile = Path.Combine(destDir, fileName);
            File.Copy(file, destFile, overwrite: true);
        }

        // Recursively copy subdirectories
        foreach (var subDir in Directory.GetDirectories(sourceDir))
        {
            var dirName = Path.GetFileName(subDir);
            var destSubDir = Path.Combine(destDir, dirName);
            CopyDirectory(subDir, destSubDir);
        }
    }
}
