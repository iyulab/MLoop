using FluentAssertions;
using Microsoft.ML;
using MLoop.Core.Pipeline;

namespace MLoop.Pipeline.Tests;

public class PipelineExecutorTests
{
    private readonly MLContext _mlContext;
    private readonly PipelineExecutor _executor;

    public PipelineExecutorTests()
    {
        _mlContext = new MLContext(seed: 42);
        _executor = new PipelineExecutor(_mlContext);
    }

    [Fact]
    public async Task ExecuteAsync_SimplePipeline_CompletesSuccessfully()
    {
        // Arrange
        var pipeline = new PipelineDefinition
        {
            Name = "Simple Test Pipeline",
            Description = "Test pipeline execution",
            Steps = new List<PipelineStep>
            {
                new()
                {
                    Name = "preprocess",
                    Type = "preprocess",
                    Parameters = new Dictionary<string, object>
                    {
                        ["input_file"] = "input.csv",
                        ["output_file"] = "output.csv"
                    },
                    ContinueOnError = false
                }
            }
        };

        // Act
        var result = await _executor.ExecuteAsync(pipeline);

        // Assert
        result.Should().NotBeNull();
        result.Status.Should().Be(PipelineStatus.Completed);
        result.PipelineName.Should().Be("Simple Test Pipeline");
        result.StepResults.Should().HaveCount(1);
        result.StepResults[0].Status.Should().Be(StepStatus.Completed);
        result.StepResults[0].StepName.Should().Be("preprocess");
        result.Duration.Should().BeGreaterThan(TimeSpan.Zero);
    }

    [Fact]
    public async Task ExecuteAsync_MultiStepPipeline_ExecutesInOrder()
    {
        // Arrange
        var pipeline = new PipelineDefinition
        {
            Name = "Multi-Step Pipeline",
            Steps = new List<PipelineStep>
            {
                new()
                {
                    Name = "preprocess_data",
                    Type = "preprocess",
                    Parameters = new Dictionary<string, object>
                    {
                        ["input_file"] = "raw.csv",
                        ["output_file"] = "clean.csv"
                    },
                    ContinueOnError = false
                },
                new()
                {
                    Name = "train_model",
                    Type = "train",
                    Parameters = new Dictionary<string, object>
                    {
                        ["data_file"] = "clean.csv",
                        ["label_column"] = "price",
                        ["training_time"] = 60
                    },
                    ContinueOnError = false
                },
                new()
                {
                    Name = "evaluate_model",
                    Type = "evaluate",
                    Parameters = new Dictionary<string, object>
                    {
                        ["model"] = "exp-001",
                        ["test_file"] = "test.csv"
                    },
                    ContinueOnError = false
                }
            }
        };

        // Act
        var result = await _executor.ExecuteAsync(pipeline);

        // Assert
        result.Status.Should().Be(PipelineStatus.Completed);
        result.StepResults.Should().HaveCount(3);
        result.StepResults.Should().OnlyContain(r => r.Status == StepStatus.Completed);

        // Verify execution order
        result.StepResults[0].StepName.Should().Be("preprocess_data");
        result.StepResults[1].StepName.Should().Be("train_model");
        result.StepResults[2].StepName.Should().Be("evaluate_model");

        // Verify timing
        result.StepResults[0].StartTime.Should().BeBefore(result.StepResults[1].StartTime);
        result.StepResults[1].StartTime.Should().BeBefore(result.StepResults[2].StartTime);
    }

    [Fact]
    public async Task ExecuteAsync_WithVariables_SubstitutesCorrectly()
    {
        // Arrange
        var pipeline = new PipelineDefinition
        {
            Name = "Variable Substitution Test",
            Variables = new Dictionary<string, object>
            {
                ["data_dir"] = "datasets",
                ["training_time"] = 120
            },
            Steps = new List<PipelineStep>
            {
                new()
                {
                    Name = "train",
                    Type = "train",
                    Parameters = new Dictionary<string, object>
                    {
                        ["data_file"] = "$data_dir",
                        ["label_column"] = "price",
                        ["training_time"] = "$training_time"
                    },
                    ContinueOnError = false
                }
            }
        };

        // Act
        var result = await _executor.ExecuteAsync(pipeline);

        // Assert
        result.Status.Should().Be(PipelineStatus.Completed);
        result.StepResults[0].Outputs.Should().ContainKey("experiment_id");
    }

    [Fact]
    public async Task ExecuteAsync_WithStepOutputChaining_PassesDataBetweenSteps()
    {
        // Arrange
        var pipeline = new PipelineDefinition
        {
            Name = "Output Chaining Test",
            Steps = new List<PipelineStep>
            {
                new()
                {
                    Name = "train_step",
                    Type = "train",
                    Parameters = new Dictionary<string, object>
                    {
                        ["data_file"] = "train.csv",
                        ["label_column"] = "price",
                        ["training_time"] = 60
                    },
                    ContinueOnError = false
                },
                new()
                {
                    Name = "evaluate_step",
                    Type = "evaluate",
                    Parameters = new Dictionary<string, object>
                    {
                        ["model"] = "$train_step.experiment_id",
                        ["test_file"] = "test.csv"
                    },
                    ContinueOnError = false
                }
            }
        };

        // Act
        var result = await _executor.ExecuteAsync(pipeline);

        // Assert
        result.Status.Should().Be(PipelineStatus.Completed);
        result.StepResults.Should().HaveCount(2);

        // Verify first step output
        result.StepResults[0].Outputs.Should().ContainKey("experiment_id");

        // Verify second step executed (proving variable substitution worked)
        result.StepResults[1].Status.Should().Be(StepStatus.Completed);
    }

    [Fact]
    public async Task ExecuteAsync_WithContinueOnError_ProceedsAfterFailure()
    {
        // Arrange - Create a pipeline with an invalid step that will fail
        var pipeline = new PipelineDefinition
        {
            Name = "Continue On Error Test",
            Steps = new List<PipelineStep>
            {
                new()
                {
                    Name = "failing_step",
                    Type = "invalid_type",  // This will fail
                    Parameters = new Dictionary<string, object>(),
                    ContinueOnError = true  // But we continue
                },
                new()
                {
                    Name = "next_step",
                    Type = "train",
                    Parameters = new Dictionary<string, object>
                    {
                        ["data_file"] = "data.csv",
                        ["label_column"] = "price",
                        ["training_time"] = 60
                    },
                    ContinueOnError = false
                }
            }
        };

        // Act
        var result = await _executor.ExecuteAsync(pipeline);

        // Assert
        result.Status.Should().Be(PipelineStatus.PartiallyCompleted);
        result.StepResults.Should().HaveCount(2);
        result.StepResults[0].Status.Should().Be(StepStatus.Failed);
        result.StepResults[1].Status.Should().Be(StepStatus.Completed);
    }

    [Fact]
    public async Task ExecuteAsync_WithoutContinueOnError_StopsAfterFailure()
    {
        // Arrange
        var pipeline = new PipelineDefinition
        {
            Name = "Stop On Error Test",
            Steps = new List<PipelineStep>
            {
                new()
                {
                    Name = "failing_step",
                    Type = "invalid_type",  // This will fail
                    Parameters = new Dictionary<string, object>(),
                    ContinueOnError = false  // Stop on error
                },
                new()
                {
                    Name = "should_not_execute",
                    Type = "train",
                    Parameters = new Dictionary<string, object>
                    {
                        ["data_file"] = "data.csv",
                        ["label_column"] = "price",
                        ["training_time"] = 60
                    },
                    ContinueOnError = false
                }
            }
        };

        // Act
        var result = await _executor.ExecuteAsync(pipeline);

        // Assert
        result.Status.Should().Be(PipelineStatus.Failed);
        result.StepResults.Should().HaveCount(1);  // Only first step executed
        result.StepResults[0].Status.Should().Be(StepStatus.Failed);
        result.Error.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public async Task ExecuteAsync_AllStepTypes_ExecuteSuccessfully()
    {
        // Arrange - Test all supported step types
        var pipeline = new PipelineDefinition
        {
            Name = "All Step Types Test",
            Steps = new List<PipelineStep>
            {
                new()
                {
                    Name = "preprocess_step",
                    Type = "preprocess",
                    Parameters = new Dictionary<string, object>
                    {
                        ["input_file"] = "raw.csv",
                        ["output_file"] = "clean.csv"
                    },
                    ContinueOnError = false
                },
                new()
                {
                    Name = "train_step",
                    Type = "train",
                    Parameters = new Dictionary<string, object>
                    {
                        ["data_file"] = "clean.csv",
                        ["label_column"] = "price",
                        ["training_time"] = 60
                    },
                    ContinueOnError = false
                },
                new()
                {
                    Name = "evaluate_step",
                    Type = "evaluate",
                    Parameters = new Dictionary<string, object>
                    {
                        ["model"] = "exp-001",
                        ["test_file"] = "test.csv"
                    },
                    ContinueOnError = false
                },
                new()
                {
                    Name = "predict_step",
                    Type = "predict",
                    Parameters = new Dictionary<string, object>
                    {
                        ["model"] = "production",
                        ["input_file"] = "predict.csv"
                    },
                    ContinueOnError = false
                },
                new()
                {
                    Name = "promote_step",
                    Type = "promote",
                    Parameters = new Dictionary<string, object>
                    {
                        ["experiment_id"] = "exp-001",
                        ["stage"] = "production"
                    },
                    ContinueOnError = false
                }
            }
        };

        // Act
        var result = await _executor.ExecuteAsync(pipeline);

        // Assert
        result.Status.Should().Be(PipelineStatus.Completed);
        result.StepResults.Should().HaveCount(5);
        result.StepResults.Should().OnlyContain(r => r.Status == StepStatus.Completed);

        // Verify each step has outputs
        result.StepResults[0].Outputs.Should().ContainKey("output_file");
        result.StepResults[1].Outputs.Should().ContainKey("experiment_id");
        result.StepResults[2].Outputs.Should().ContainKey("r_squared");
        result.StepResults[3].Outputs.Should().ContainKey("prediction_file");
        result.StepResults[4].Outputs.Should().ContainKey("promoted");
    }

    [Fact]
    public async Task ExecuteAsync_EmptyPipeline_CompletesSuccessfully()
    {
        // Arrange
        var pipeline = new PipelineDefinition
        {
            Name = "Empty Pipeline",
            Steps = new List<PipelineStep>()
        };

        // Act
        var result = await _executor.ExecuteAsync(pipeline);

        // Assert
        result.Status.Should().Be(PipelineStatus.Completed);
        result.StepResults.Should().BeEmpty();
        result.Duration.Should().BeGreaterThanOrEqualTo(TimeSpan.Zero);
    }

    [Fact]
    public async Task ExecuteAsync_WithLogger_LogsProgress()
    {
        // Arrange
        var logs = new List<string>();
        var pipeline = new PipelineDefinition
        {
            Name = "Logging Test Pipeline",
            Description = "Tests logging functionality",
            Steps = new List<PipelineStep>
            {
                new()
                {
                    Name = "test_step",
                    Type = "train",
                    Parameters = new Dictionary<string, object>
                    {
                        ["data_file"] = "data.csv",
                        ["label_column"] = "price",
                        ["training_time"] = 60
                    },
                    ContinueOnError = false
                }
            }
        };

        // Act
        var result = await _executor.ExecuteAsync(pipeline, log => logs.Add(log));

        // Assert
        result.Status.Should().Be(PipelineStatus.Completed);
        logs.Should().NotBeEmpty();
        logs.Should().Contain(l => l.Contains("Starting pipeline"));
        logs.Should().Contain(l => l.Contains("Logging Test Pipeline"));
        logs.Should().Contain(l => l.Contains("Step 1/1"));
        logs.Should().Contain(l => l.Contains("completed"));
    }

    [Fact]
    public async Task ExecuteAsync_WithCancellation_StopsExecution()
    {
        // Arrange
        var cts = new CancellationTokenSource();
        var pipeline = new PipelineDefinition
        {
            Name = "Cancellation Test",
            Steps = new List<PipelineStep>
            {
                new()
                {
                    Name = "step1",
                    Type = "train",
                    Parameters = new Dictionary<string, object>
                    {
                        ["data_file"] = "data.csv",
                        ["label_column"] = "price",
                        ["training_time"] = 60
                    },
                    ContinueOnError = false
                }
            }
        };

        // Cancel immediately
        cts.Cancel();

        // Act & Assert
        await _executor.Invoking(e => e.ExecuteAsync(pipeline, cancellationToken: cts.Token))
            .Should().ThrowAsync<OperationCanceledException>();
    }

    [Fact]
    public async Task ExecuteAsync_TimingMetrics_AreAccurate()
    {
        // Arrange
        var pipeline = new PipelineDefinition
        {
            Name = "Timing Test",
            Steps = new List<PipelineStep>
            {
                new()
                {
                    Name = "step1",
                    Type = "train",
                    Parameters = new Dictionary<string, object>
                    {
                        ["data_file"] = "data.csv",
                        ["label_column"] = "price",
                        ["training_time"] = 60
                    },
                    ContinueOnError = false
                }
            }
        };

        // Act
        var result = await _executor.ExecuteAsync(pipeline);

        // Assert
        result.StartTime.Should().BeBefore(result.EndTime);
        result.Duration.Should().Be(result.EndTime - result.StartTime);

        foreach (var stepResult in result.StepResults)
        {
            stepResult.StartTime.Should().BeBefore(stepResult.EndTime);
            stepResult.Duration.Should().Be(stepResult.EndTime - stepResult.StartTime);
            stepResult.StartTime.Should().BeOnOrAfter(result.StartTime);
            stepResult.EndTime.Should().BeOnOrBefore(result.EndTime);
        }
    }

    [Fact]
    public async Task ExecuteAsync_ParallelSteps_ExecuteConcurrently()
    {
        // Arrange
        var pipeline = new PipelineDefinition
        {
            Name = "Parallel Execution Test",
            Steps = new List<PipelineStep>
            {
                new()
                {
                    Name = "parallel_step_1",
                    Type = "train",
                    Parameters = new Dictionary<string, object>
                    {
                        ["data_file"] = "data1.csv",
                        ["label_column"] = "price",
                        ["training_time"] = 60
                    },
                    ContinueOnError = false,
                    Parallel = true
                },
                new()
                {
                    Name = "parallel_step_2",
                    Type = "train",
                    Parameters = new Dictionary<string, object>
                    {
                        ["data_file"] = "data2.csv",
                        ["label_column"] = "price",
                        ["training_time"] = 60
                    },
                    ContinueOnError = false,
                    Parallel = true
                },
                new()
                {
                    Name = "sequential_step",
                    Type = "evaluate",
                    Parameters = new Dictionary<string, object>
                    {
                        ["model"] = "exp-001",
                        ["test_file"] = "test.csv"
                    },
                    ContinueOnError = false,
                    DependsOn = new List<string> { "parallel_step_1", "parallel_step_2" }
                }
            }
        };

        // Act
        var result = await _executor.ExecuteAsync(pipeline);

        // Assert
        result.Status.Should().Be(PipelineStatus.Completed);
        result.StepResults.Should().HaveCount(3);
        result.StepResults.Should().OnlyContain(r => r.Status == StepStatus.Completed);

        // Verify parallel steps started at similar times (within 100ms)
        var parallelSteps = result.StepResults.Take(2).ToList();
        var timeDiff = Math.Abs((parallelSteps[0].StartTime - parallelSteps[1].StartTime).TotalMilliseconds);
        timeDiff.Should().BeLessThan(100);

        // Verify sequential step started after parallel steps completed
        var sequentialStep = result.StepResults[2];
        sequentialStep.StartTime.Should().BeOnOrAfter(parallelSteps[0].EndTime);
        sequentialStep.StartTime.Should().BeOnOrAfter(parallelSteps[1].EndTime);
    }

    [Fact]
    public async Task ExecuteAsync_DependencyChain_RespectsOrder()
    {
        // Arrange
        var pipeline = new PipelineDefinition
        {
            Name = "Dependency Chain Test",
            Steps = new List<PipelineStep>
            {
                new()
                {
                    Name = "step_a",
                    Type = "preprocess",
                    Parameters = new Dictionary<string, object>
                    {
                        ["input_file"] = "raw.csv",
                        ["output_file"] = "clean.csv"
                    },
                    ContinueOnError = false
                },
                new()
                {
                    Name = "step_b",
                    Type = "train",
                    Parameters = new Dictionary<string, object>
                    {
                        ["data_file"] = "clean.csv",
                        ["label_column"] = "price",
                        ["training_time"] = 60
                    },
                    ContinueOnError = false,
                    DependsOn = new List<string> { "step_a" }
                },
                new()
                {
                    Name = "step_c",
                    Type = "evaluate",
                    Parameters = new Dictionary<string, object>
                    {
                        ["model"] = "exp-001",
                        ["test_file"] = "test.csv"
                    },
                    ContinueOnError = false,
                    DependsOn = new List<string> { "step_b" }
                }
            }
        };

        // Act
        var result = await _executor.ExecuteAsync(pipeline);

        // Assert
        result.Status.Should().Be(PipelineStatus.Completed);
        result.StepResults[0].StepName.Should().Be("step_a");
        result.StepResults[1].StepName.Should().Be("step_b");
        result.StepResults[2].StepName.Should().Be("step_c");

        // Verify execution order
        result.StepResults[1].StartTime.Should().BeOnOrAfter(result.StepResults[0].EndTime);
        result.StepResults[2].StartTime.Should().BeOnOrAfter(result.StepResults[1].EndTime);
    }
}
