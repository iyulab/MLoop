"""Tests for `run.py regress` - the rule is pre-registered in the function's docstring; these make sure it catches a built flip and ignores noise.

  python -m unittest benchmarks/engine-compare/test_regress.py     (needs numpy)
"""
import contextlib, importlib.util, io, json, os, random, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('engine_run', os.path.join(HERE, 'run.py'))
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def write(rows):
    handle = tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False)
    for r in rows:
        handle.write(json.dumps(r) + '\n')
    handle.close()
    return handle.name


def rows(dataset, task, mloop_gap, noise, seeds=range(5), rng=None):
    """Regression rows (lower RMSE is better): the reference gets 1.0 plus its own noise, MLoop gets the reference plus `mloop_gap` (> 0 = worse)."""
    out = []
    for s in seeds:
        ref = 1.0 + rng.gauss(0, noise)
        out.append({'dataset': dataset, 'task': task, 'rows': 1000, 'seed': s, 'system': 'flaml', 'test': {'rmse': ref}})
        out.append({'dataset': dataset, 'task': task, 'rows': 1000, 'seed': s, 'system': 'mloop', 'test': {'rmse': ref + mloop_gap + rng.gauss(0, noise / 4)}})
    return out


def judge(base, cand):
    with contextlib.redirect_stdout(io.StringIO()):
        return run.regress([write(base)], [write(cand)])


class RegressRule(unittest.TestCase):
    def test_a_built_flip_is_caught(self):
        rng = random.Random(1)
        names = [f't{i}' for i in range(8)]
        base = [r for n in names for r in rows(n, 'regression', -0.1, 0.02, rng=rng)]
        # three tables go from clearly ahead of the reference to clearly behind it
        cand = [r for i, n in enumerate(names) for r in rows(n, 'regression', 0.2 if i < 3 else -0.1, 0.02, rng=rng)]
        self.assertEqual('REGRESSION', judge(base, cand))

    def test_a_new_draw_of_noise_around_the_same_gaps_is_ok(self):
        # Tables whose gap is clear stay where they are whatever the seeds draw. (A table on the boundary between equal and behind can flip
        # on noise alone - that is why one flip is only WATCH and a regression needs a rise of two.)
        for seed in range(20):
            rng = random.Random(seed)
            names = [f't{i}' for i in range(8)]
            base = [r for n in names for r in rows(n, 'regression', -0.1, 0.05, rng=rng)]
            cand = [r for n in names for r in rows(n, 'regression', -0.1, 0.05, rng=rng)]
            self.assertEqual('OK', judge(base, cand), f'seed {seed}')

    def test_one_flip_is_only_watched(self):
        rng = random.Random(3)
        names = [f't{i}' for i in range(6)]
        base = [r for n in names for r in rows(n, 'regression', -0.1, 0.02, rng=rng)]
        cand = [r for i, n in enumerate(names) for r in rows(n, 'regression', 0.3 if i == 0 else -0.1, 0.02, rng=rng)]
        self.assertEqual('WATCH', judge(base, cand))

    def test_a_machine_that_is_slower_for_both_systems_is_not_a_regression(self):
        # the runner shifts both systems alike: the reference carries the machine, so the verdict does not move
        rng = random.Random(4)
        names = [f't{i}' for i in range(6)]
        base = [r for n in names for r in rows(n, 'regression', -0.1, 0.02, rng=rng)]
        cand = []
        for r in (r for n in names for r in rows(n, 'regression', -0.1, 0.02, rng=rng)):
            r['test']['rmse'] += 0.5     # a worse machine for everybody: both systems lose the same
            cand.append(r)
        self.assertEqual('OK', judge(base, cand))


if __name__ == '__main__':
    unittest.main()
