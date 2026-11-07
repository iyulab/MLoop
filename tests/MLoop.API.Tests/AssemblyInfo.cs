using Xunit;
using Xunit.Sdk;
using Xunit.v3;

// A test here sets MLOOP_MODEL_NAME, a variable the whole process reads: the classes of this assembly run one after another
// (xunit.runner.json says so as well; this is the declaration the contract test looks for).
[assembly: Parallelization(Mode = ParallelMode.None)]
