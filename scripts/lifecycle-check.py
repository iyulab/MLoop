#!/usr/bin/env python3
"""Walk a model through its operating life with the real CLI and say, step by step, whether each stage behaved.

One scenario per task, on generated tables (standard library only; nothing is downloaded) - or, with --tables, one per public
table cut into two regimes by lifecycle-tables.py (the label depends on the same walk, the shifted rows are a real part of the
population and their label is rewritten):

  1  init                     a project is created for the task
  2  train                    a first model is trained and, being the first, promoted
  3  status                   the project names it as production
  4  predict --log            predictions on in-regime rows are answered and logged with ids
  5  drift (control)          in-regime rows show no meaningful drift
  6  drift (shifted inputs)   rows whose inputs have moved are flagged, on the column that moved
  7  feedback + trigger       ground truth for the shifted rows is recorded; the retraining trigger fires
  8  retrain, no gate          a model trained on the new regime is the better one there, yet the recorded scores (each its own model's
                              held-out score, on different data) keep it in staging - and the reason says how to compare on the same rows
  9  paired evaluation        `evaluate --against production` prefers it on the new regime's held-out rows
 10  explicit promote         the production slot moves to it and the history records the change and the gate
 11  rollback                 the slot returns to the model it replaced and the history records that
 12  retrain, paired gate     with gate.data (current-regime rows) and gate.no_worse_than_production, the same retraining is promoted by
                              the row-by-row comparison and the history records the gate as passed
 13+ serve                   `mloop serve` over HTTP: /health, /predict (logged, with ids), /logs, /feedback, /drift, /trigger, an admin /promote
                              that the next /predict sees (the model cache reloads), /rollback, /promotions, and an async /train job
                              that finishes and reports whether it was promoted;
                           then under load: 16 clients at once answer exactly what a lone client gets and are logged once each with
                           distinct ids, a promotion and a rollback made while clients keep predicting never produce an error, and
                           --soak SECONDS holds a steady load and compares the server's memory at the start and the end, and
                           --bench SECONDS prints a throughput / latency table by clients and rows per request

  python lifecycle-check.py --mloop "dotnet tools/MLoop.CLI/bin/Release/net10.0/mloop.dll" [--tasks binary,multiclass,regression]
         [--time 60] [--trials 10] [--out lifecycle-results.json] [--keep]
  python lifecycle-check.py --mloop "..." --tables DIR [--datasets credit-g,houses]    # DIR as written by lifecycle-tables.py

Exit status is 1 when a step fails that is not named in --expected-failures. A step named there is still run and
reported ("xfail"); it turning green is reported as "xpass" so the list can be shortened. Each run is a fresh
temporary project, removed unless --keep is given.
"""
import argparse, csv, json, math, os, random, shlex, shutil, signal, socket, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

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


def to_number(text):
    """A cell as a JSON number; an empty cell is a missing value (null), as the loaders read it."""
    return float(text) if text.strip() else None


def write_table(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x1', 'x2', 'x3', 'cat', 'y'])
        w.writerows(rows)


class Run:
    def __init__(self, mloop, task, work, budget, expected, trials=0):
        self.mloop, self.task, self.work, self.budget, self.expected = shlex.split(mloop), task, work, budget, expected
        self.trials = trials
        self.steps = []

    def budget_args(self):
        """The training budget: a trial count when there is one (the same search on any machine), the time then only its ceiling."""
        return ['--time', str(self.budget)] + (['--max-trials', str(self.trials)] if self.trials else [])

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


def free_port():
    with socket.socket() as sk:
        sk.bind(('127.0.0.1', 0))
        return sk.getsockname()[1]


def http(method, url, token=None, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    if token:
        req.add_header('Authorization', f'Bearer {token}')
    if data is not None:
        req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw.strip() else None)
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, raw
    except (urllib.error.URLError, OSError) as e:
        return 0, str(e)


def server_rss_mb(server_pid):
    """Resident memory of the server: `mloop serve` starts the API as a child, so the launcher alone says nothing — the whole process group."""
    pids = subprocess.run(['pgrep', '-g', str(server_pid)], capture_output=True, text=True).stdout.split()
    if not pids:
        return -1
    out = subprocess.run(['ps', '-o', 'rss=', '-p', ','.join(pids)], capture_output=True, text=True).stdout.split()
    return sum(int(x) for x in out) / 1024


def serve_bench(run, url, admin, feats, server_pid, seconds):
    """Throughput and latency of POST /predict by number of clients and rows per request: numbers to read, not a bound to meet.

    The clients are Python threads in one process, so the high-concurrency rows are a lower bound of what the server does.
    """
    rss_mb = lambda: server_rss_mb(server_pid)
    lines, errors_total = [], 0
    for rows_per_request in (1, 100):
        body = (feats * (rows_per_request // len(feats) + 1))[:rows_per_request]
        for clients in (1, 4, 16, 64):
            stop, latencies, errors = threading.Event(), [], []
            def work():
                while not stop.is_set():
                    t0 = time.time()
                    c, _ = http('POST', url + '/predict?name=default', admin, body)
                    latencies.append(time.time() - t0)
                    if c != 200:
                        errors.append(c)
            threads = [threading.Thread(target=work) for _ in range(clients)]
            start = time.time()
            for th in threads:
                th.start()
            time.sleep(seconds)
            stop.set()
            for th in threads:
                th.join()
            elapsed = time.time() - start
            latencies.sort()
            n = len(latencies)
            p = lambda q: latencies[min(n - 1, int(n * q))] * 1000 if n else 0
            errors_total += len(errors)
            lines.append(f'rows={rows_per_request:>3} clients={clients:>2}: {n / elapsed:6.0f} req/s {n * rows_per_request / elapsed:7.0f} rows/s p50={p(.5):5.0f}ms p99={p(.99):5.0f}ms errors={len(errors)}')
    run.check('serve_bench', errors_total == 0, f'rss={rss_mb():.0f}MB | ' + ' | '.join(lines))


def serve_load_steps(run, url, admin, feats, before, cand, server_pid, soak, proj=None, bench=0):
    """The service under load: many clients at once, a promotion made while they predict, and (with --soak) a steady load held."""
    def predict():
        return http('POST', url + '/predict?name=default', admin, feats)

    def strip(p):
        return {k: v for k, v in p.items() if k != 'predictionId'}

    code, base = predict()
    want = [strip(p) for p in (base or {}).get('predictions', [])] if isinstance(base, dict) else []
    _, logs0 = http('GET', url + '/logs?name=default&limit=1000000', admin)
    n0 = len((logs0 or {}).get('logs', [])) if isinstance(logs0, dict) else -1

    workers, per_worker = 16, 10
    def burst(_):
        return [predict() for _ in range(per_worker)]
    with ThreadPoolExecutor(workers) as ex:
        results = [r for chunk in ex.map(burst, range(workers)) for r in chunk]
    bad = [c for c, _ in results if c != 200]
    answered = [r for c, r in results if c == 200 and isinstance(r, dict)]
    differ = sum(1 for r in answered if [strip(p) for p in r.get('predictions', [])] != want)
    ids = [p.get('predictionId') for r in answered for p in r.get('predictions', [])]
    expected_rows = workers * per_worker * len(feats)
    run.check('serve_concurrent_predict', not bad and differ == 0 and len(ids) == expected_rows and len(set(ids)) == len(ids),
              f'requests={len(results)} non200={len(bad)} differing_answers={differ} ids={len(ids)}/{expected_rows} distinct={len(set(ids))}')
    _, logs1 = http('GET', url + '/logs?name=default&limit=1000000', admin)
    n1 = len((logs1 or {}).get('logs', [])) if isinstance(logs1, dict) else -1
    run.check('serve_concurrent_logged', n0 >= 0 and n1 - n0 == expected_rows, f'logged {n1 - n0} of {expected_rows} predicted rows')

    # A promotion and a rollback while clients keep predicting: no request fails, every answer is by the old or the new model.
    stop, errors, answers = threading.Event(), [], []
    def hammer():
        while not stop.is_set():
            c, r = predict()
            if c != 200:
                errors.append(c)
            elif isinstance(r, dict):
                answers.append(r.get('experimentId'))
    # What the production directory looked like whenever the model file was not there: a failed request's cause, seen from outside.
    production = os.path.join(proj, 'models', 'default', 'production') if proj else None
    absences = []
    def watch():
        while not stop.is_set():
            if production and not os.path.exists(os.path.join(production, 'model.zip')):
                absences.append((round(time.time() % 1000, 3), sorted(os.listdir(production)) if os.path.isdir(production) else 'no production dir'))
            time.sleep(0.0005)
    threads = [threading.Thread(target=hammer) for _ in range(8)] + [threading.Thread(target=watch)]
    for th in threads:
        th.start()
    time.sleep(2)
    pc, _ = http('POST', url + '/promote', admin, {'name': 'default', 'experimentId': cand})
    time.sleep(2)
    rc, _ = http('POST', url + '/rollback', admin, {'name': 'default'})
    time.sleep(2)
    stop.set()
    for th in threads:
        th.join()
    _, last = predict()
    last_by = last.get('experimentId') if isinstance(last, dict) else None
    run.check('serve_promote_under_load', pc == 200 and rc == 200 and not errors and set(answers) <= {before, cand} and cand in set(answers) and last_by == before,
              f'promote={pc} rollback={rc} requests={len(answers) + len(errors)} errors={errors[:3]} answered_by={sorted(set(map(str, answers)))} after_rollback={last_by}'
              + (f' model_file_absent={len(absences)}x first={absences[0]}' if absences else ''))

    if soak > 0:
        rss_mb = lambda: server_rss_mb(server_pid)
        stop, errors, latencies = threading.Event(), [], []
        def steady():
            while not stop.is_set():
                t0 = time.time()
                c, _ = predict()
                latencies.append(time.time() - t0)
                if c != 200:
                    errors.append(c)
                time.sleep(0.05)
        threads = [threading.Thread(target=steady) for _ in range(4)]
        for th in threads:
            th.start()
        series, start = [], time.time()
        while time.time() - start < soak:
            time.sleep(max(1, soak / 6))
            series.append(round(rss_mb()))
        stop.set()
        for th in threads:
            th.join()
        warm = series[1] if len(series) > 1 else series[0]
        grew = (series[-1] - warm) / warm if warm > 0 else 0
        latencies.sort()
        p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0
        run.check('serve_soak', not errors and 0 < series[-1] and grew <= 0.25,
                  f'{soak}s requests={len(latencies)} errors={len(errors)} rss_mb={series} growth_after_warmup={grew:+.0%} p99={p99 * 1000:.0f}ms')

    if bench > 0:
        serve_bench(run, url, admin, feats, server_pid, bench)


SERVER_LOG = None   # --server-log: where the server's output goes (otherwise discarded)


def serve_steps(run, proj, base, task, budget, table, soak=0, bench=0):
    """The service half of the life: the same project, driven over HTTP by a server started on a free port."""
    rc, out, err = run.cli('token', '--role', 'admin', '-q', cwd=proj)
    admin = out.strip()
    port = free_port()
    url = f'http://127.0.0.1:{port}'
    # The request limiter is off here: these steps measure the server under load, not the limiter (one caller, hundreds of requests a minute).
    server = subprocess.Popen(run.mloop + ['serve', '--port', str(port), '--host', '127.0.0.1'], cwd=proj,
                              env={**os.environ, 'Serve__RateLimit__PermitsPerMinute': '0'},
                              stdout=(open(SERVER_LOG, 'ab') if SERVER_LOG else subprocess.DEVNULL),
                              stderr=subprocess.STDOUT if SERVER_LOG else subprocess.DEVNULL, start_new_session=True)
    try:
        up = False
        for _ in range(60):
            code, _ = http('GET', url + '/health', timeout=2)
            if code == 200:
                up = True
                break
            time.sleep(1)
        run.check('serve_health', up and bool(admin), f'port={port} up={up}')
        if not up:
            return
        code, info = http('GET', url + '/info?name=default', admin)
        info = info if isinstance(info, dict) else {}
        before = info.get('experimentId') or info.get('productionExperimentId')
        rows = list(csv.DictReader(open(os.path.join(base, 'new_a.csv'))))[:5]
        feats = [{k: (to_number(v) if k in table['numeric_columns'] else v) for k, v in r.items() if k != table['label']} for r in rows]
        code, pr = http('POST', url + '/predict?name=default', admin, feats)
        preds = pr.get('predictions') if isinstance(pr, dict) else None
        run.check('serve_predict', code == 200 and preds is not None and len(preds) == 5 and all(p.get('predictionId') for p in preds)
                  and pr.get('experimentId') == before, f'code={code} n={len(preds or [])} answered_by={pr.get("experimentId") if isinstance(pr, dict) else pr} production={before}')
        code, logs = http('GET', url + '/logs?name=default&limit=1000', admin)
        entries = (logs or {}).get('logs', []) if isinstance(logs, dict) else []
        run.check('serve_logs', code == 200 and len(entries) >= 5, f'code={code} entries={len(entries)}')
        if preds:
            code, _ = http('POST', url + '/feedback', admin, {'predictionId': preds[0]['predictionId'], 'modelName': 'default',
                                                             'actualValue': str(rows[0][table['label']]), 'source': 'lifecycle-check'})
        else:
            code = 0
        run.check('serve_feedback', code in (200, 201), f'code={code}')
        code, dr = http('GET', url + '/drift?name=default', admin)
        run.check('serve_drift', code == 200 and isinstance(dr, dict), f'code={code}')
        code, tr = http('GET', url + '/trigger?name=default', admin)
        run.check('serve_trigger', code == 200 and isinstance(tr, dict) and 'shouldRetrain' in tr, f'code={code}')

        cand = 'exp-002'
        serve_load_steps(run, url, admin, feats, before, cand, server.pid, soak, proj, bench)

        # A promotion made while the server holds the model in its cache must reach the next answer.
        code, _ = http('POST', url + '/promote', admin, {'name': 'default', 'experimentId': cand})
        code2, pr2 = http('POST', url + '/predict?name=default', admin, feats)
        run.check('serve_promote_reloads_cache', code == 200 and isinstance(pr2, dict) and pr2.get('experimentId') == cand,
                  f'promote={code} answered_by={pr2.get("experimentId") if isinstance(pr2, dict) else pr2}')
        code, _ = http('POST', url + '/rollback', admin, {'name': 'default'})
        code2, pr3 = http('POST', url + '/predict?name=default', admin, feats)
        run.check('serve_rollback_reloads_cache', code == 200 and isinstance(pr3, dict) and pr3.get('experimentId') == before,
                  f'rollback={code} answered_by={pr3.get("experimentId") if isinstance(pr3, dict) else pr3}')
        code, hist = http('GET', url + '/promotions?name=default', admin)
        acts = [h.get('action') for h in ((hist or {}).get('promotions') or [])] if isinstance(hist, dict) else []
        run.check('serve_promotions', code == 200 and 'rollback' in acts, f'code={code} actions={acts}')

        # The server reads data from inside the project only, relative paths being the project's.
        os.makedirs(os.path.join(proj, 'datasets'), exist_ok=True)
        shutil.copy(os.path.join(base, 'train_b.csv'), os.path.join(proj, 'datasets', 'train_b.csv'))
        code, job = http('POST', url + '/train', admin, {'dataFile': 'datasets/train_b.csv', 'name': 'default', 'labelColumn': table['label'],
                                                        'task': TASK_FLAG[task], 'timeLimitSeconds': budget, 'promote': False, **({'maxTrials': run.trials} if run.trials else {})})
        job_id = (job.get('jobId') or job.get('id')) if isinstance(job, dict) else None
        final = {}
        deadline = time.time() + budget * 6 + 60
        while job_id and time.time() < deadline:
            _, final = http('GET', f'{url}/jobs/{job_id}', admin)
            if isinstance(final, dict) and str(final.get('status', '')).lower() in ('completed', 'failed'):
                break
            time.sleep(2)
        status = str(final.get('status', '')).lower() if isinstance(final, dict) else ''
        run.check('serve_train_job', code == 202 and status == 'completed' and final.get('promoted') is False,
                  f'accepted={code} status={status} promoted={final.get("promoted") if isinstance(final, dict) else None}')
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


def generated_tables(task, base):
    """The standard-library tables: a label that depends on x1 as it sits in the training regime. Returns the table's description."""
    rng = random.Random(20261029)
    for name, rows in [('train_a', make_table(task, 2000, rng)), ('new_a', make_table(task, 300, rng)),
                       ('train_b', make_table(task, 2000, rng, SHIFT)), ('test_b', make_table(task, 600, rng, SHIFT)),
                       ('new_b', make_table(task, 300, rng, SHIFT))]:
        write_table(os.path.join(base, name + '.csv'), rows)
    return {'name': task, 'task': task, 'label': 'y', 'shift_column': 'x1', 'numeric_columns': ['x1', 'x2', 'x3'],
            'n_new_a': 300, 'n_new_b': 300, 'learned': list(LEARNED[task]), 'public': False}


def public_tables(src, base):
    """One table cut by lifecycle-tables.py: copy its five files and read its description."""
    for name in ('train_a', 'new_a', 'train_b', 'test_b', 'new_b'):
        shutil.copy(os.path.join(src, name + '.csv'), os.path.join(base, name + '.csv'))
    desc = json.load(open(os.path.join(src, 'manifest.json')))
    return {**desc, 'public': True}


def scenario(mloop, task, budget, keep, expected, serve, soak=0, trials=0, bench=0, src=None):
    base = tempfile.mkdtemp(prefix=f'lifecycle-{task}-')
    run = Run(mloop, task, base, budget, expected, trials)
    table = public_tables(src, base) if src else generated_tables(task, base)
    label, shift_col, n_new_a = table['label'], table['shift_column'], table['n_new_a']
    run.task = table['name']  # a step is reported under the table it ran on
    print(f'== {table["name"]}  ({base})', flush=True)
    try:
        new_b = list(csv.DictReader(open(os.path.join(base, 'new_b.csv'))))
        proj = os.path.join(base, 'proj')

        rc, init = run.json_of('init', 'proj', '--task', TASK_FLAG[task], '--label', label, '--json')
        run.check('init', rc == 0 and init and init.get('task') == TASK_FLAG[task], f'rc={rc}')

        rc, evs = run.events('train', os.path.join(base, 'train_a.csv'), '--label', label, *run.budget_args(), '--json', cwd=proj)
        result = next((e for e in evs if e.get('event') == 'result'), {})
        promo = next((e for e in evs if e.get('event') == 'promotion'), {})
        key, floor = table['learned']
        score = (result.get('metrics') or {}).get(key)
        run.check('train', rc == 0 and result.get('experimentId') == 'exp-001' and promo.get('promoted') is True
                  and score is not None and score >= floor, f'{key}={score} promoted={promo.get("promoted")}')

        rc, status = run.json_of('status', '--json', cwd=proj)
        models = (status or {}).get('models') or [{}]
        run.check('status', models[0].get('productionExperimentId') == 'exp-001', f'production={models[0].get("productionExperimentId")}')

        rc, pa = run.json_of('predict', os.path.join(base, 'new_a.csv'), '--log', '--json', cwd=proj)
        preds = (pa or {}).get('predictions') or []
        run.check('predict_log', rc == 0 and len(preds) == n_new_a and all(p.get('predictionId') for p in preds), f'{len(preds)} predictions')

        rc, drift = run.json_of('drift', '--json', cwd=proj)
        level = ((drift or {}).get('drift') or {}).get('level')
        # Generated tables are steady, so nothing moves. On a real table with many columns and a few hundred rows, ten bins per column
        # leave a column's PSI above 0.1 now and then by chance; the control only has to stay below the significance threshold.
        max_psi = ((drift or {}).get('drift') or {}).get('maxPsi', 0)
        run.check('drift_control', rc == 0 and (level == 'None' or (table['public'] and level == 'Moderate' and max_psi < 0.25)),
                  f'level={level} maxPsi={max_psi:.3f}')

        rc, pb = run.json_of('predict', os.path.join(base, 'new_b.csv'), '--log', '--json', cwd=proj)
        preds_b = (pb or {}).get('predictions') or []
        rc, drift = run.json_of('drift', '--json', cwd=proj)
        d = (drift or {}).get('drift') or {}
        ranked = sorted(d.get('features') or [], key=lambda f: -f.get('psi', 0))
        moved = (ranked[0] if ranked else {}).get('column')
        # On generated tables one column moves and it is the most moved. On a real table the columns that go with it move too, so the
        # column cut on only has to be flagged as significantly moved itself (the product's own threshold, PSI 0.25).
        by_column = {f.get('column'): f.get('psi', 0) for f in ranked}
        flagged = shift_col in by_column and by_column[shift_col] >= 0.25 if table['public'] else moved == shift_col
        run.check('drift_shifted', d.get('level') == 'Significant' and flagged,
                  f'level={d.get("level")} most moved={moved} (cut on {shift_col}, psi {by_column.get(shift_col, 0):.2f})')

        want = min(50, table['n_new_b'], len(preds_b))
        given = 0
        for i, p in enumerate(preds_b[:want]):
            rc, _, _ = run.cli('feedback', 'add', '--prediction-id', p['predictionId'], '--actual-value', str(new_b[i][label]), '--source', 'lifecycle-check', cwd=proj)
            given += rc == 0
        rc, trig = run.json_of('trigger', 'check', '--feedback', str(min(30, want)), '--json', cwd=proj)
        run.check('trigger', rc == 0 and given == want and (trig or {}).get('shouldRetrain') is True,
                  f'feedback={given} shouldRetrain={(trig or {}).get("shouldRetrain")}')

        rc, evs = run.events('train', os.path.join(base, 'train_b.csv'), '--label', label, *run.budget_args(), '--json', cwd=proj)
        result = next((e for e in evs if e.get('event') == 'result'), {})
        promo = next((e for e in evs if e.get('event') == 'promotion'), {})
        candidate = result.get('experimentId')
        reason = promo.get('keptInStagingReason') or ''
        # Without a gate the recorded scores decide, as they always have: each model's own held-out score, on its own data. On generated tables
        # the new regime scores no higher, so the candidate stays in staging and the reason says how to compare on the same rows. On a real
        # table the new regime can be the easier one and its model out-scores production's record - then it is promoted, which is the rule
        # working as written; the slot is put back so the paired steps that follow have a production to compare against.
        if rc == 0 and promo.get('promoted') is True and table['public']:
            run.check('retrain_without_gate', True, f'{candidate} promoted by its recorded score (the new regime scored higher)')
            rc, rb = run.json_of('rollback', '--json', cwd=proj)
            run.check('rollback_of_auto_promotion', (rb or {}).get('status') == 'rolled-back' and (rb or {}).get('experimentId') == 'exp-001',
                      f'status={(rb or {}).get("status")} to={(rb or {}).get("experimentId")}')
        else:
            run.check('retrain_without_gate', rc == 0 and promo.get('promoted') is False and '--against production' in reason,
                      f'{candidate} promoted={promo.get("promoted")} reason={reason[:90]}')

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

        # The project says how a challenger is judged: against production, row by row, on rows of the regime it serves now.
        config_path = os.path.join(proj, 'mloop.yaml')
        text = open(config_path).read()
        marker = '    description: '
        at = text.index(marker)
        eol = text.index('\n', at) + 1
        gate = f'    gate:\n      data: {os.path.join(base, "test_b.csv")}\n      no_worse_than_production: true\n'
        open(config_path, 'w').write(text[:eol] + gate + text[eol:])
        rc, evs = run.events('train', os.path.join(base, 'train_b.csv'), '--label', label, *run.budget_args(), '--json', cwd=proj)
        result = next((e for e in evs if e.get('event') == 'result'), {})
        promo = next((e for e in evs if e.get('event') == 'promotion'), {})
        rc2, hist = run.json_of('promote', '--history', '--json', cwd=proj)
        changes = (hist or {}).get('promotions') or []
        last = changes[-1] if changes else {}
        run.check('retrain_with_paired_gate', rc == 0 and promo.get('promoted') is True and last.get('experimentId') == result.get('experimentId')
                  and last.get('gate') == 'passed', f'{result.get("experimentId")} promoted={promo.get("promoted")} reason={promo.get("keptInStagingReason")} history last={last.get("experimentId")}/{last.get("gate")}')

        if serve:
            serve_steps(run, proj, base, task, budget, table, soak, bench)
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
    ap.add_argument('--tables', default='', help='a folder written by lifecycle-tables.py: one scenario per public table in it, instead of the generated tables')
    ap.add_argument('--datasets', default='', help='with --tables: comma-separated table names (default: every table in the folder)')
    ap.add_argument('--time', type=int, default=60, help='training time ceiling in seconds')
    ap.add_argument('--trials', type=int, default=10, help='trials of the trainer search (the same search on any machine, so a step does not turn on a lucky draw); 0 = the time budget alone')
    ap.add_argument('--out', default='lifecycle-results.json')
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('--soak', type=int, default=0, help='seconds of steady load against the server, comparing its memory at the start and the end (0 = skip)')
    ap.add_argument('--bench', type=int, default=0, help='seconds per cell of a throughput / latency table for POST /predict by clients (1, 4, 16, 64) and rows per request (1, 100); 0 = skip')
    ap.add_argument('--server-log', default='', help='append the server\'s output to this file (default: discarded)')
    ap.add_argument('--no-serve', action='store_true', help='skip the HTTP steps (they start a server on a free local port)')
    ap.add_argument('--expected-failures', default='', help='comma-separated step names that are known to fail')
    args = ap.parse_args()
    global SERVER_LOG
    SERVER_LOG = args.server_log or None
    expected = {s for s in args.expected_failures.split(',') if s}
    steps = []
    if args.tables:
        wanted = set(args.datasets.split(',')) - {''}
        for name in sorted(os.listdir(args.tables)):
            src = os.path.join(args.tables, name)
            if not os.path.exists(os.path.join(src, 'manifest.json')) or (wanted and name not in wanted):
                continue
            task = json.load(open(os.path.join(src, 'manifest.json')))['task']
            found = scenario(args.mloop, task, args.time, args.keep, expected, not args.no_serve, args.soak, args.trials, args.bench, src)
            steps += [{**st, 'table': name, 'task': task} for st in found]
    else:
        for task in args.tasks.split(','):
            steps += scenario(args.mloop, task, args.time, args.keep, expected, not args.no_serve, args.soak, args.trials, args.bench)
    with open(args.out, 'w') as f:
        json.dump(steps, f, indent=2)
    tally = {}
    for s in steps:
        tally[s['status']] = tally.get(s['status'], 0) + 1
    print('\n' + ', '.join(f'{k} {v}' for k, v in sorted(tally.items())))
    sys.exit(1 if tally.get('FAIL') else 0)


if __name__ == '__main__':
    main()
