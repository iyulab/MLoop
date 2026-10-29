#!/usr/bin/env python3
"""Walk a model through its operating life with the real CLI and say, step by step, whether each stage behaved.

One scenario per task, on generated tables (standard library only; nothing is downloaded):

  1  init                     a project is created for the task
  2  train                    a first model is trained and, being the first, promoted
  3  status                   the project names it as production
  4  predict --log            predictions on in-regime rows are answered and logged with ids
  5  drift (control)          in-regime rows show no meaningful drift
  6  drift (shifted inputs)   rows whose inputs have moved are flagged, on the column that moved
  7  feedback + trigger       ground truth for the shifted rows is recorded; the retraining trigger fires
  8  retrain on the new regime  a model trained on the new regime is promoted when it is the better model
  9  paired evaluation        `evaluate --against production` prefers it on the new regime's held-out rows
 10  explicit promote         the production slot moves to it and the history records the change and the gate
 11  rollback                 the slot returns to the model it replaced and the history records that

  python lifecycle-check.py --mloop "dotnet tools/MLoop.CLI/bin/Release/net10.0/mloop.dll" [--tasks binary,multiclass,regression]
         [--time 15] [--out lifecycle-results.json] [--keep]

Exit status is 1 when a step fails that is not named in --expected-failures. A step named there is still run and
reported ("xfail"); it turning green is reported as "xpass" so the list can be shortened. Each run is a fresh
temporary project, removed unless --keep is given.
"""
import argparse, csv, json, math, os, random, shlex, shutil, subprocess, sys, tempfile, time

TASK_FLAG = {'binary': 'binary-classification', 'multiclass': 'multiclass-classification', 'regression': 'regression'}
# the metric a first model must reach to count as having learned the table (loose: this checks the wiring, not accuracy)
LEARNED = {'binary': ('accuracy', 0.8), 'multiclass': ('macro_accuracy', 0.6), 'regression': ('r_squared', 0.7)}
SHIFT = 2.5


def make_table(task, n, rng, shift=0.0):
    """Rows whose label depends on x1 as it sits in the training regime: a shifted x1 reads as a different answer."""
    rows = []
    for _ in range(n):
        x1 = rng.gauss(0, 1) + shift
        x2, x3, cat = rng.gauss(0, 1), rng.random(), rng.choice('abc')
        z = 1.5 * (x1 - shift) - 1.0 * x2 + (0.8 if cat == 'a' else -0.4 if cat == 'b' else 0.0) + rng.gauss(0, 0.7)
        if task == 'binary':
            y = 1 if z > 0 else 0
        elif task == 'multiclass':
            y = 0 if z < -1.0 else 1 if z < 1.0 else 2
        else:
            y = round(3.0 * z + 10.0, 4)
        rows.append([round(x1, 4), round(x2, 4), round(x3, 4), cat, y])
    return rows


def write_table(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x1', 'x2', 'x3', 'cat', 'y'])
        w.writerows(rows)


class Run:
    def __init__(self, mloop, task, work, budget, expected):
        self.mloop, self.task, self.work, self.budget, self.expected = shlex.split(mloop), task, work, budget, expected
        self.steps = []

    def cli(self, *args, cwd=None):
        p = subprocess.run(self.mloop + list(args), cwd=cwd or self.work, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr

    def json_of(self, *args, cwd=None):
        rc, out, err = self.cli(*args, cwd=cwd)
        try:
            return rc, json.loads(out)
        except ValueError:
            return rc, None

    def events(self, *args, cwd=None):
        rc, out, err = self.cli(*args, cwd=cwd)
        evs = []
        for line in out.splitlines():
            line = line.strip()
            if line.startswith('{'):
                try:
                    evs.append(json.loads(line))
                except ValueError:
                    pass
        return rc, evs

    def check(self, step, ok, detail=''):
        name = f'{self.task}/{step}'
        known = step in self.expected
        status = ('xpass' if ok else 'xfail') if known else ('pass' if ok else 'FAIL')
        self.steps.append({'step': name, 'status': status, 'detail': detail})
        print(f'  [{status:>5}] {step}: {detail}', flush=True)
        return ok


def scenario(mloop, task, budget, keep, expected):
    base = tempfile.mkdtemp(prefix=f'lifecycle-{task}-')
    run = Run(mloop, task, base, budget, expected)
    rng = random.Random(20261029)
    print(f'== {task}  ({base})', flush=True)
    try:
        for name, rows in [('train_a', make_table(task, 2000, rng)), ('new_a', make_table(task, 300, rng)),
                           ('train_b', make_table(task, 2000, rng, SHIFT)), ('test_b', make_table(task, 600, rng, SHIFT)),
                           ('new_b', make_table(task, 300, rng, SHIFT))]:
            write_table(os.path.join(base, name + '.csv'), rows)
        new_b = list(csv.DictReader(open(os.path.join(base, 'new_b.csv'))))
        proj = os.path.join(base, 'proj')

        rc, init = run.json_of('init', 'proj', '--task', TASK_FLAG[task], '--label', 'y', '--json')
        run.check('init', rc == 0 and init and init.get('task') == TASK_FLAG[task], f'rc={rc}')

        rc, evs = run.events('train', os.path.join(base, 'train_a.csv'), '--label', 'y', '--time', str(budget), '--json', cwd=proj)
        result = next((e for e in evs if e.get('event') == 'result'), {})
        promo = next((e for e in evs if e.get('event') == 'promotion'), {})
        key, floor = LEARNED[task]
        score = (result.get('metrics') or {}).get(key)
        run.check('train', rc == 0 and result.get('experimentId') == 'exp-001' and promo.get('promoted') is True
                  and score is not None and score >= floor, f'{key}={score} promoted={promo.get("promoted")}')

        rc, status = run.json_of('status', '--json', cwd=proj)
        models = (status or {}).get('models') or [{}]
        run.check('status', models[0].get('productionExperimentId') == 'exp-001', f'production={models[0].get("productionExperimentId")}')

        rc, pa = run.json_of('predict', os.path.join(base, 'new_a.csv'), '--log', '--json', cwd=proj)
        preds = (pa or {}).get('predictions') or []
        run.check('predict_log', rc == 0 and len(preds) == 300 and all(p.get('predictionId') for p in preds), f'{len(preds)} predictions')

        rc, drift = run.json_of('drift', '--json', cwd=proj)
        level = ((drift or {}).get('drift') or {}).get('level')
        run.check('drift_control', rc == 0 and level == 'None', f'level={level}')

        rc, pb = run.json_of('predict', os.path.join(base, 'new_b.csv'), '--log', '--json', cwd=proj)
        preds_b = (pb or {}).get('predictions') or []
        rc, drift = run.json_of('drift', '--json', cwd=proj)
        d = (drift or {}).get('drift') or {}
        moved = max(d.get('features') or [{}], key=lambda f: f.get('psi', 0), default={}).get('column')
        run.check('drift_shifted', d.get('level') == 'Significant' and moved == 'x1', f'level={d.get("level")} most moved={moved}')

        given = 0
        for i, p in enumerate(preds_b[:50]):
            rc, _, _ = run.cli('feedback', 'add', '--prediction-id', p['predictionId'], '--actual-value', str(new_b[i]['y']), '--source', 'lifecycle-check', cwd=proj)
            given += rc == 0
        rc, trig = run.json_of('trigger', 'check', '--feedback', '30', '--json', cwd=proj)
        run.check('trigger', rc == 0 and given == 50 and (trig or {}).get('shouldRetrain') is True,
                  f'feedback={given} shouldRetrain={(trig or {}).get("shouldRetrain")}')

        rc, evs = run.events('train', os.path.join(base, 'train_b.csv'), '--label', 'y', '--time', str(budget), '--json', cwd=proj)
        result = next((e for e in evs if e.get('event') == 'result'), {})
        promo = next((e for e in evs if e.get('event') == 'promotion'), {})
        candidate = result.get('experimentId')
        # The new model is the better one on the data the service now sees (step 9 measures that); the loop is meant to put it in production.
        run.check('retrain_promotes', rc == 0 and promo.get('promoted') is True,
                  f'{candidate} promoted={promo.get("promoted")} reason={promo.get("keptInStagingReason")}')

        rc, ev = run.json_of('evaluate', candidate or 'exp-002', os.path.join(base, 'test_b.csv'), '--against', 'production', '--json', cwd=proj)
        verdict = (((ev or {}).get('comparison') or {}).get('paired') or {}).get('verdict')
        run.check('paired_evaluation', rc == 0 and verdict == 'CandidateBetter', f'verdict={verdict}')

        rc, pr = run.json_of('promote', candidate or 'exp-002', '--json', cwd=proj)
        rc2, hist = run.json_of('promote', '--history', '--json', cwd=proj)
        changes = (hist or {}).get('promotions') or []
        last = changes[-1] if changes else {}
        run.check('explicit_promote', (pr or {}).get('status') in ('promoted', 'already-production') and last.get('experimentId') == candidate
                  and last.get('gate') == 'passed', f'status={(pr or {}).get("status")} history={len(changes)} last={last.get("experimentId")}/{last.get("gate")}')

        rc, rb = run.json_of('rollback', '--json', cwd=proj)
        rc2, hist = run.json_of('promote', '--history', '--json', cwd=proj)
        changes = (hist or {}).get('promotions') or []
        run.check('rollback', (rb or {}).get('status') == 'rolled-back' and (rb or {}).get('experimentId') == 'exp-001'
                  and changes and changes[-1].get('experimentId') == 'exp-001', f'status={(rb or {}).get("status")} to={(rb or {}).get("experimentId")}')
    except Exception as ex:  # a step that cannot run is a failed scenario, not a crashed harness
        run.check('scenario', False, f'{type(ex).__name__}: {ex}')
    finally:
        if keep:
            print(f'   kept {base}')
        else:
            shutil.rmtree(base, ignore_errors=True)
    return run.steps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mloop', required=True, help='command that starts the mloop CLI, e.g. "dotnet path/to/mloop.dll"')
    ap.add_argument('--tasks', default='binary,multiclass,regression')
    ap.add_argument('--time', type=int, default=15, help='training budget in seconds')
    ap.add_argument('--out', default='lifecycle-results.json')
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('--expected-failures', default='', help='comma-separated step names (e.g. retrain_promotes) that are known to fail')
    args = ap.parse_args()
    expected = {s for s in args.expected_failures.split(',') if s}
    steps = []
    for task in args.tasks.split(','):
        steps += scenario(args.mloop, task, args.time, args.keep, expected)
    with open(args.out, 'w') as f:
        json.dump(steps, f, indent=2)
    tally = {}
    for s in steps:
        tally[s['status']] = tally.get(s['status'], 0) + 1
    print('\n' + ', '.join(f'{k} {v}' for k, v in sorted(tally.items())))
    sys.exit(1 if tally.get('FAIL') else 0)


if __name__ == '__main__':
    main()
