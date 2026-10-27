using Microsoft.ML.OnnxRuntime;

namespace MLoop.Core.Embedding;

/// <summary>
/// The one place an ONNX Runtime session's options are made. A session here serves a caller that, on the same
/// machine, uses the CPU as soon as the answer arrives (an embedding feeds a nearest-neighbour search) — so a worker
/// thread must not keep the core after the run is over.
/// </summary>
internal static class OnnxSessionOptions
{
    /// <summary>
    /// ONNX Runtime warns, on stderr, about shape metadata it merges leniently (ML.NET's converter writes it for every
    /// one-hot column) — noise to a user, so only errors surface. Intra-op threads do not spin while idle: spinning
    /// trades the CPU after a run for a lower wake-up latency, which a server answering a few requests a second does not need.
    /// </summary>
    public static SessionOptions Create()
    {
        var options = new SessionOptions { LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR };
        options.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
        return options;
    }
}
