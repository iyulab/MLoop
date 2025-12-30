// Copyright (c) IYULab. All rights reserved.
// Licensed under the MIT License.

using Ironbees.Core.Goals;
using MLoop.AIAgent.Core.Rules;
using MLoop.AIAgent.Core.Sampling;

namespace MLoop.AIAgent.Core.Integration;

/// <summary>
/// Adapts ironbees AgenticSettings to MLoop.AIAgent components.
/// Bridges the declarative schema from ironbees with MLoop's execution logic.
/// </summary>
public static class AgenticSettingsAdapter
{
    /// <summary>
    /// Maps ironbees SamplingStrategy enum to MLoop SamplingMethod.
    /// </summary>
    public static SamplingMethod ToSamplingMethod(this Ironbees.Core.Goals.SamplingStrategy strategy)
    {
        return strategy switch
        {
            Ironbees.Core.Goals.SamplingStrategy.Progressive => SamplingMethod.Stratified, // Progressive uses stratified with stages
            Ironbees.Core.Goals.SamplingStrategy.Random => SamplingMethod.Random,
            Ironbees.Core.Goals.SamplingStrategy.Stratified => SamplingMethod.Stratified,
            Ironbees.Core.Goals.SamplingStrategy.Sequential => SamplingMethod.Systematic,
            _ => SamplingMethod.Stratified
        };
    }

    /// <summary>
    /// Creates a SamplingManager from ironbees SamplingSettings.
    /// </summary>
    public static SamplingManager CreateSamplingManager(SamplingSettings? settings, int? seed = null)
    {
        var manager = new SamplingManager(seed);

        if (settings != null)
        {
            manager.Method = settings.Strategy.ToSamplingMethod();
        }

        return manager;
    }

    /// <summary>
    /// Configures a ConfidenceCalculator from ironbees ConfidenceSettings.
    /// </summary>
    public static void Configure(this ConfidenceCalculator calculator, ConfidenceSettings? settings)
    {
        if (settings == null) return;

        calculator.StabilityThreshold = settings.Threshold;

        // Map StabilityWindow to ConvergenceSampleCount
        // StabilityWindow is iterations, we convert to approximate sample count
        calculator.ConvergenceSampleCount = settings.StabilityWindow * 100;
    }

    /// <summary>
    /// Creates a ConfidenceCalculator configured from ironbees ConfidenceSettings.
    /// </summary>
    public static ConfidenceCalculator CreateConfidenceCalculator(ConfidenceSettings? settings)
    {
        var calculator = new ConfidenceCalculator();
        calculator.Configure(settings);
        return calculator;
    }

    /// <summary>
    /// Determines if HITL should be triggered based on ironbees HitlSettings.
    /// </summary>
    public static bool ShouldTriggerHitl(
        HitlSettings? settings,
        double currentConfidence,
        string? checkpoint = null,
        bool hasException = false)
    {
        if (settings == null)
            return false;

        return settings.Policy switch
        {
            HitlPolicy.Always => true,
            HitlPolicy.Never => false,
            HitlPolicy.OnUncertainty => currentConfidence < settings.UncertaintyThreshold,
            HitlPolicy.OnException => hasException,
            HitlPolicy.OnThreshold => currentConfidence < settings.UncertaintyThreshold,
            _ => false
        } || (checkpoint != null && settings.Checkpoints.Contains(checkpoint));
    }

    /// <summary>
    /// Gets the timeout action as a string description.
    /// </summary>
    public static string GetTimeoutActionDescription(HitlSettings? settings)
    {
        if (settings == null)
            return "Pause and wait for response";

        return settings.TimeoutAction switch
        {
            HitlTimeoutAction.Pause => "Pause and wait for response",
            HitlTimeoutAction.ContinueWithDefault => "Continue with default action",
            HitlTimeoutAction.Cancel => "Cancel the workflow",
            HitlTimeoutAction.Skip => "Skip this checkpoint",
            _ => "Pause and wait for response"
        };
    }

    /// <summary>
    /// Calculates stage configurations based on ironbees SamplingSettings.
    /// </summary>
    public static StageSamplingConfig[] CreateStageConfigs(SamplingSettings? settings)
    {
        if (settings == null)
        {
            // Use default progressive stages
            return SamplingManager.StageConfigs;
        }

        var initialSize = settings.InitialBatchSize;
        var growthFactor = settings.GrowthFactor;
        var maxSamples = settings.MaxSamples ?? int.MaxValue;

        // Generate stage configs based on growth factor
        var stages = new List<StageSamplingConfig>();
        var currentSize = initialSize;

        for (int stage = 1; stage <= 4; stage++)
        {
            var purpose = stage switch
            {
                1 => "Initial Exploration",
                2 => "Pattern Expansion",
                3 => "HITL Decision",
                4 => "Confidence Checkpoint",
                _ => "Processing"
            };

            // Convert absolute size to approximate rate
            var rate = Math.Min(1.0, currentSize / 100000.0); // Assume 100K baseline

            stages.Add(new StageSamplingConfig(
                stage,
                rate,
                Math.Min(currentSize, maxSamples),
                purpose));

            currentSize = (int)(currentSize * growthFactor);
        }

        // Stage 5 is always full processing
        stages.Add(new StageSamplingConfig(5, 1.0, int.MaxValue, "Bulk Processing"));

        return [.. stages];
    }

    /// <summary>
    /// Validates that AgenticSettings are coherent.
    /// </summary>
    public static List<string> Validate(AgenticSettings? settings)
    {
        var errors = new List<string>();

        if (settings == null)
            return errors;

        // Validate Sampling
        if (settings.Sampling != null)
        {
            if (settings.Sampling.InitialBatchSize <= 0)
                errors.Add("SamplingSettings.InitialBatchSize must be positive");

            if (settings.Sampling.GrowthFactor <= 0)
                errors.Add("SamplingSettings.GrowthFactor must be positive");

            if (settings.Sampling.MinSamplesForConfidence < 0)
                errors.Add("SamplingSettings.MinSamplesForConfidence cannot be negative");
        }

        // Validate Confidence
        if (settings.Confidence != null)
        {
            if (settings.Confidence.Threshold < 0 || settings.Confidence.Threshold > 1)
                errors.Add("ConfidenceSettings.Threshold must be between 0 and 1");

            if (settings.Confidence.StabilityWindow <= 0)
                errors.Add("ConfidenceSettings.StabilityWindow must be positive");

            if (settings.Confidence.MinConfidenceForHitl.HasValue &&
                (settings.Confidence.MinConfidenceForHitl < 0 || settings.Confidence.MinConfidenceForHitl > 1))
                errors.Add("ConfidenceSettings.MinConfidenceForHitl must be between 0 and 1");
        }

        // Validate HITL
        if (settings.Hitl != null)
        {
            if (settings.Hitl.UncertaintyThreshold < 0 || settings.Hitl.UncertaintyThreshold > 1)
                errors.Add("HitlSettings.UncertaintyThreshold must be between 0 and 1");
        }

        return errors;
    }
}
