"""Tests for what makes a result row reproducible: the locked comparison libraries, the provenance stamped on every row, and the licence of every table.

  python -m unittest benchmarks/engine-compare/test_repro.py     (needs numpy, as run.py does)

A row that names no versions cannot be reproduced or questioned once the CI log that made it is gone, so these keep three things true:
the workflows install from the lock (not `pip install flaml ...` at whatever version is newest that day), the lock pins every line, and a row
carries what produced it.
"""
import importlib.util, json, os, re, sys, unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
WORKFLOWS = os.path.join(HERE, '..', '..', '.github', 'workflows')
spec = importlib.util.spec_from_file_location('engine_run', os.path.join(HERE, 'run.py'))
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

# libraries a sweep compares against; installing one of them without a version is what the lock exists to prevent
COMPARED = ('flaml', 'lightgbm', 'xgboost', 'scikit-learn', 'pandas', 'numpy', 'autogluon')
# workflows that are not part of the engine comparison (they check one library against its own reference)
OUT_OF_SCOPE = {'lgbm-parity.yml'}


def unlocked_installs(text):
    """Lines of a workflow that `pip install` a compared library by name, with neither a requirements file (-r) nor a constraint file (-c)."""
    found = []
    for line in text.splitlines():
        stripped = line.strip()
        match = re.search(r'\bpip install\b(.*)', stripped)
        if not match:
            continue
        arguments = match.group(1)
        if re.search(r'(^|\s)-[rc]\s', arguments):
            continue
        if any(re.search(r'(^|[\s"\'])' + re.escape(name) + r'([\s"\'\[=<>.]|$)', arguments) for name in COMPARED):
            found.append(stripped)
    return found


def pinned(path):
    """(name, version) for every requirement line of a compiled lock."""
    out = []
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        out.append(tuple(line.split('==')) if '==' in line else (line, None))
    return out


class WorkflowsInstallFromTheLock(unittest.TestCase):
    def test_no_workflow_installs_a_compared_library_unpinned(self):
        offenders = {}
        scanned = 0
        for name in sorted(os.listdir(WORKFLOWS)):
            if not name.endswith('.yml') or name in OUT_OF_SCOPE:
                continue
            scanned += 1
            bad = unlocked_installs(open(os.path.join(WORKFLOWS, name), encoding='utf-8').read())
            if bad:
                offenders[name] = bad
        self.assertGreater(scanned, 0)
        self.assertEqual({}, offenders, 'a workflow installs a compared library at whatever version is newest; install from requirements.txt')

    def test_the_scan_sees_the_shape_it_forbids_and_lets_the_legitimate_ones_through(self):
        self.assertEqual(['run: pip install flaml lightgbm xgboost scikit-learn pandas numpy'],
                         unlocked_installs('      run: pip install flaml lightgbm xgboost scikit-learn pandas numpy\n'))
        self.assertEqual(['run: pip install "autogluon.tabular[lightgbm,catboost,xgboost]"'],
                         unlocked_installs('      run: pip install "autogluon.tabular[lightgbm,catboost,xgboost]"\n'))
        self.assertEqual(['run: pip install numpy'], unlocked_installs('      run: pip install numpy\n'))
        self.assertEqual([], unlocked_installs('      run: pip install -r benchmarks/engine-compare/requirements.txt\n'))
        self.assertEqual([], unlocked_installs('      run: pip install -c benchmarks/engine-compare/requirements.txt numpy\n'))
        self.assertEqual([], unlocked_installs('      run: pip install pytest\n'))   # not a compared library

    def test_the_workflows_that_sweep_do_name_the_lock(self):
        # the scan above passes for a workflow that installs nothing at all, so the two sweeps must be seen reading the lock
        for name in ('engine-compare.yml', 'benchmarks-scheduled.yml'):
            text = open(os.path.join(WORKFLOWS, name), encoding='utf-8').read()
            self.assertIn('benchmarks/engine-compare/requirements.txt', text, name)


class TheLockPinsEverything(unittest.TestCase):
    def test_every_requirement_line_is_an_exact_pin_and_the_compared_libraries_are_in(self):
        for file, required in (('requirements.txt', ('flaml', 'lightgbm', 'xgboost', 'scikit-learn', 'pandas', 'numpy')),
                               ('requirements-autogluon.txt', ('autogluon-tabular', 'flaml', 'lightgbm', 'xgboost', 'scikit-learn', 'pandas', 'numpy'))):
            lines = pinned(os.path.join(HERE, file))
            self.assertGreater(len(lines), len(required), file)
            self.assertEqual([], [name for name, version in lines if version is None], f'{file}: a line without ==')
            names = {name for name, _ in lines}
            for name in required:
                self.assertIn(name, names, f'{file} lacks {name}')

    def test_a_requirement_without_a_version_is_seen(self):
        with mock.patch('builtins.open', mock.mock_open(read_data='numpy\nflaml==2.8.0\n# comment\n')):
            self.assertEqual([('numpy', None), ('flaml', '2.8.0')], pinned('x'))


class ARowCarriesWhatMadeIt(unittest.TestCase):
    ENTRY = {'id': 1, 'name': 't', 'task': 'binary', 'target': 'y', 'openml_licence': 'CC BY 4.0'}

    def test_every_table_names_the_licence_openml_declares(self):
        entries = json.load(open(os.path.join(HERE, 'datasets.json'), encoding='utf-8'))
        self.assertGreaterEqual(len(entries), 16)
        self.assertEqual([], [e['name'] for e in entries if not e.get('openml_licence')])

    def test_provenance_names_the_versions_the_interpreter_and_the_ci_run(self):
        env = {'GITHUB_SHA': 'abc123', 'GITHUB_RUN_ID': '42', 'GITHUB_RUN_ATTEMPT': '2'}
        with mock.patch.dict(os.environ, env):
            stamp = run.provenance([sys.executable, '-c', 'import sys; print("0.0.0-test")'])
        self.assertEqual('0.0.0-test', stamp['mloop'])
        self.assertEqual(('abc123', '42', '2'), (stamp['commit'], stamp['run_id'], stamp['run_attempt']))
        self.assertEqual('.'.join(map(str, sys.version_info[:3])), stamp['python'])
        self.assertIn('numpy', stamp['libraries'])   # this suite needs numpy, so the stamp must see it

    def test_a_cli_that_cannot_start_leaves_the_version_unknown_not_the_sweep_stopped(self):
        stamp = run.provenance(['/no/such/mloop'])
        self.assertIsNone(stamp['mloop'])
        self.assertIsInstance(stamp['libraries'], dict)

    def test_the_row_keeps_its_old_fields_and_adds_the_stamp(self):
        stamp = {'mloop': '1.2.3', 'libraries': {'flaml': '2.8.0'}}
        row = run.result_row(self.ENTRY, 100, 3, 'mloop', 60, {'test': {'auc': 0.9}}, stamp)
        self.assertEqual({'dataset': 't', 'openml_id': 1, 'task': 'binary', 'rows': 100, 'seed': 3, 'system': 'mloop', 'budget': 60,
                          'test': {'auc': 0.9}}, {k: v for k, v in row.items() if k not in ('openml_licence', 'provenance')})
        self.assertEqual('CC BY 4.0', row['openml_licence'])
        self.assertEqual(stamp, row['provenance'])
        json.dumps(row)   # a row is one line of a jsonl file


if __name__ == '__main__':
    unittest.main()
