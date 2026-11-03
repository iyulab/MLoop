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

  python lifecycle-check.py --mloop "dotnet tools/MLoop.CLI/bin/Release/net10.0/mloop.dll" [--tasks binary,multiclass,regression,text,forecast,anomaly,clustering,outliers,ranking,recommendation]
         [--time 60] [--trials 10] [--out lifecycle-results.json] [--keep]
  python lifecycle-check.py --mloop "..." --tables DIR [--datasets credit-g,houses]    # DIR as written by lifecycle-tables.py

On a table the first model learns nothing from (the promotion gate keeps it out of production), the later steps cannot run and the table is reported once as "skip", not as failures.
Exit status is 1 when a step fails that is not named in --expected-failures. A step named there is still run and
reported ("xfail"); it turning green is reported as "xpass" so the list can be shortened. Each run is a fresh
temporary project, removed unless --keep is given.
"""
import argparse, csv, json, math, os, random, shlex, shutil, signal, socket, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

TASK_FLAG = {'binary': 'binary-classification', 'multiclass': 'multiclass-classification', 'regression': 'regression', 'text': 'text-classification', 'forecast': 'forecasting', 'anomaly': 'time-series-anomaly',
             'clustering': 'clustering', 'outliers': 'anomaly-detection',
             'ranking': 'ranking', 'recommendation': 'recommendation'}
# the metric a first model must reach to count as having learned the table (loose: this checks the wiring, not accuracy)
LEARNED = {'binary': ('accuracy', 0.8), 'multiclass': ('macro_accuracy', 0.6), 'regression': ('r_squared', 0.7), 'text': ('macro_accuracy', 0.6)}
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


# a text table's two regimes use different words for the same two answers: a model that learned the first regime reads the second as noise
TEXT_FILLER = ['the', 'a', 'it', 'this', 'was', 'is', 'very', 'product', 'item', 'really', 'and', 'but']
TEXT_WORDS = {False: {'pos': ['great', 'love', 'excellent', 'fast', 'works', 'happy', 'recommend'], 'neg': ['broken', 'slow', 'refund', 'bad', 'late', 'awful', 'waste']},
              True: {'pos': ['superb', 'delightful', 'flawless', 'sturdy', 'pleased', 'bargain', 'reliable'], 'neg': ['faulty', 'cracked', 'overpriced', 'flimsy', 'defective', 'useless', 'regret']}}


def make_text_table(n, rng, shifted=False):
    """Rows of a review and its answer; the shifted regime says it with other words."""
    rows = []
    for _ in range(n):
        positive = rng.random() < 0.5
        vocabulary = TEXT_WORDS[shifted]
        signal = [rng.choice(vocabulary['pos'] if (positive if rng.random() < 0.85 else not positive) else vocabulary['neg']) for _ in range(3)]
        words = [rng.choice(TEXT_FILLER) for _ in range(5)] + signal
        rng.shuffle(words)
        rows.append([' '.join(words), 'pos' if positive else 'neg'])
    return rows


def to_number(text):
    """A cell as a JSON number; an empty cell is a missing value (null), as the loaders read it."""
    return float(text) if text.strip() else None


def write_table(path, rows, header=('x1', 'x2', 'x3', 'cat', 'y')):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(list(header))
        w.writerows(rows)


class Run:
    def __init__(self, mloop, task, work, budget, expected, trials=0):
        self.mloop, self.task, self.work, self.budget, self.expected = shlex.split(mloop), task, work, budget, expected
        # a deep-learning task trains by epochs and refuses a trial count
        self.trials = 0 if task in ('text', 'forecast', 'anomaly', 'clustering', 'outliers', 'ranking', 'recommendation') else trials
        self.metric = []   # extra `train` arguments that name what the table is judged on
        self.last_output = ''   # the tail of the last command's output, shown with a failed step
        self.steps = []

    def budget_args(self):
        """The training budget: a trial count when there is one (the same search on any machine), the time then only its ceiling."""
        return ['--time', str(self.budget)] + (['--max-trials', str(self.trials)] if self.trials else []) + self.metric

    def cli(self, *args, cwd=None):
        p = subprocess.run(self.mloop + list(args), cwd=cwd or self.work, capture_output=True, text=True)
        self.last_output = (p.stdout + p.stderr)[-600:].strip()
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

    def skip(self, step, detail=''):
        """A premise of the walk is not met (it needs a promoted first model and the table gave none): reported, never counted as a failure."""
        self.steps.append({'step': f'{self.task}/{step}', 'status': 'skip', 'detail': detail})
        print(f'  [ skip] {step}: {detail}', flush=True)

    def check(self, step, ok, detail=''):
        name = f'{self.task}/{step}'
        if not ok and self.last_output:
            detail = f'{detail} | last output: {self.last_output}'
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
        rows = read_rows(os.path.join(base, 'new_a.csv'), table)[:5]
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
    if task == 'text':
        for name, rows in [('train_a', make_text_table(1500, rng)), ('new_a', make_text_table(300, rng)), ('train_b', make_text_table(1500, rng, True)),
                           ('test_b', make_text_table(500, rng, True)), ('new_b', make_text_table(300, rng, True))]:
            write_table(os.path.join(base, name + '.csv'), rows, ('review', 'y'))
        return {'name': task, 'task': task, 'label': 'y', 'shift_column': 'review', 'numeric_columns': [], 'n_new_a': 300, 'n_new_b': 300,
                'learned': list(LEARNED[task]), 'public': False}
    for name, rows in [('train_a', make_table(task, 2000, rng)), ('new_a', make_table(task, 300, rng)),
                       ('train_b', make_table(task, 2000, rng, SHIFT)), ('test_b', make_table(task, 600, rng, SHIFT)),
                       ('new_b', make_table(task, 300, rng, SHIFT))]:
        write_table(os.path.join(base, name + '.csv'), rows)
    return {'name': task, 'task': task, 'label': 'y', 'shift_column': 'x1', 'numeric_columns': ['x1', 'x2', 'x3'],
            'n_new_a': 300, 'n_new_b': 300, 'learned': list(LEARNED[task]), 'public': False}


def read_rows(path, table):
    """The rows of one of the table's files as the commands read them: in the table's encoding, without the unnamed index column
    of a spreadsheet export, a header cell broken over two lines as one name with a space."""
    with open(path, newline='', encoding=table.get('encoding', 'utf-8')) as f:
        return [{k.replace('\r\n', ' ').replace('\n', ' '): v for k, v in r.items() if k} for r in csv.DictReader(f)]


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
    if table['public'] and task == 'binary':
        # a public table is rarely balanced; judged on accuracy (the default) a model that learned something can sit at the majority share
        # and be refused by the promotion gate, which says nothing about the lifecycle
        run.metric = ['--metric', 'auc']
    print(f'== {table["name"]}  ({base})', flush=True)
    try:
        new_b = read_rows(os.path.join(base, 'new_b.csv'), table)
        proj = os.path.join(base, 'proj')

        rc, init = run.json_of('init', 'proj', '--task', TASK_FLAG[task], '--label', label, '--json')
        run.check('init', rc == 0 and init and init.get('task') == TASK_FLAG[task], f'rc={rc}')

        if table.get('expect') == 'refused':
            # a table the right answer to which is a refusal (lifecycle-tables.py --synthetic duplicate-name): training says why, by
            # the name of the column, before it spends the search, and leaves no experiment behind — nothing later has a model to walk
            rc, out, err = run.cli('train', os.path.join(base, 'train_a.csv'), '--label', label, *run.budget_args(), cwd=proj)
            run.check('train_is_refused', rc != 0 and table.get('refusal_mentions', '') in out + err, f'rc={rc} {(out + err).strip()[-160:]}')
            rc, status = run.json_of('status', '--json', cwd=proj)
            models = (status or {}).get('models') or [{}]
            run.check('refusal_leaves_no_model', not models[0].get('productionExperimentId'), f'production={models[0].get("productionExperimentId")}')
            return run.steps

        rc, evs = run.events('train', os.path.join(base, 'train_a.csv'), '--label', label, *run.budget_args(), '--json', cwd=proj)
        result = next((e for e in evs if e.get('event') == 'result'), {})
        promo = next((e for e in evs if e.get('event') == 'promotion'), {})
        key, floor = table['learned']
        score = (result.get('metrics') or {}).get(key)
        learned = rc == 0 and result.get('experimentId') == 'exp-001' and promo.get('promoted') is True and score is not None and score >= floor
        reason = promo.get('keptInStagingReason') or ''
        if not learned and table['public'] and rc == 0 and promo.get('promoted') is False and 'quality gate' in reason:
            # A table the first model learned nothing from is refused by the promotion gate — the right answer — and every later step
            # needs a model in production. That is a premise of the walk not met, not eighteen failures: say so once and stop.
            run.skip('train', f'{key}={score}: the table taught the first model nothing in this budget, so the quality gate kept it out '
                              f'of production; the walk needs a promoted first model | reason: {reason}')
            return run.steps
        run.check('train', learned, f'{key}={score} promoted={promo.get("promoted")}')

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
        # a text column is profiled by descriptors, reported as 'review (unseen words)': the column cut on is the one named first
        flagged = shift_col in by_column and by_column[shift_col] >= 0.25 if table['public'] else (moved or '').split(' (')[0] == shift_col
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


# A series has one column and no rows to ask about, so its walk is its own: the model replays the file it was trained on and answers with the
# horizon after the file's last row. Two regimes follow one another on one clock (t): the second has another level, swing and period, so a
# model trained on the first forecasts the second's continuation badly.
FORECAST_HORIZON = 10


def make_series(t0, n, regime, rng):
    # the second regime has another period: the recurrence a season of 12 teaches does not forecast a season of 7
    level, swing, noise, period = (100.0, 10.0, 1.0, 12) if regime == 'a' else (140.0, 25.0, 2.0, 7)
    return [[round(level + 0.05 * t + swing * math.sin(2 * math.pi * t / period) + rng.gauss(0, noise), 4)] for t in range(t0, t0 + n)]


def forecast_error(preds, actual_rows):
    """Mean absolute error of a horizon forecast (the rows of `predict --json`) against the values that followed."""
    if len(preds) != len(actual_rows):
        return None   # an empty or short forecast is not a good one
    return sum(abs(p['score'] - float(a[0])) for p, a in zip(preds, actual_rows)) / len(actual_rows)


def forecast_serve_steps(run, proj, new_a, new_b):
    """The service half of a series' life: the forecast over HTTP, a candidate forecast beside production, a promotion that the cache sees."""
    rc, out, err = run.cli('token', '--role', 'admin', '-q', cwd=proj)
    admin = out.strip()
    port = free_port()
    url = f'http://127.0.0.1:{port}'
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

        code, fc = http('POST', url + '/predict?name=default', admin, {})
        rows = (fc or {}).get('predictions') if isinstance(fc, dict) else None
        err_now = forecast_error(rows or [], new_b)
        run.check('serve_forecast', code == 200 and isinstance(fc, dict) and fc.get('task') == 'forecasting' and fc.get('experimentId') == before
                  and fc.get('count') == FORECAST_HORIZON and err_now is not None and err_now < 5,
                  f'code={code} answered_by={fc.get("experimentId") if isinstance(fc, dict) else fc} production={before} error against the values that followed={err_now}')

        code, bad = http('POST', url + '/predict?name=default', admin, {'horizon': FORECAST_HORIZON + 7})
        run.check('serve_forecast_other_horizon_is_refused', code == 400 and 'horizon' in json.dumps(bad), f'code={code} {json.dumps(bad)[:140]}')

        # a candidate answers beside production, from its own series: the old regime's continuation
        code, cand = http('POST', url + '/predict?name=default&experiment=exp-001', admin, {})
        crows = (cand or {}).get('predictions') if isinstance(cand, dict) else None
        err_cand = forecast_error(crows or [], new_a)
        run.check('serve_candidate_forecast', code == 200 and isinstance(cand, dict) and cand.get('experimentId') == 'exp-001' and err_cand is not None and err_cand < 6,
                  f'code={code} answered_by={cand.get("experimentId") if isinstance(cand, dict) else cand} error={err_cand}')

        # the project's gate judges on the stretch production serves now: the old regime's model is worse there and is refused
        code, refusal = http('POST', url + '/promote', admin, {'name': 'default', 'experimentId': 'exp-001'})
        code1, pr1 = http('POST', url + '/predict?name=default', admin, {})
        run.check('serve_gate_refuses_the_stale_candidate', code == 422 and isinstance(pr1, dict) and pr1.get('experimentId') == before,
                  f'promote={code} answered_by={pr1.get("experimentId") if isinstance(pr1, dict) else pr1} {json.dumps(refusal)[:120]}')
        code, _ = http('POST', url + '/promote', admin, {'name': 'default', 'experimentId': 'exp-001', 'force': True})
        code2, pr2 = http('POST', url + '/predict?name=default', admin, {})
        run.check('serve_promote_reloads_cache', code == 200 and isinstance(pr2, dict) and pr2.get('experimentId') == 'exp-001',
                  f'promote={code} answered_by={pr2.get("experimentId") if isinstance(pr2, dict) else pr2}')
        code, _ = http('POST', url + '/rollback', admin, {'name': 'default'})
        code2, pr3 = http('POST', url + '/predict?name=default', admin, {})
        run.check('serve_rollback_reloads_cache', code == 200 and isinstance(pr3, dict) and pr3.get('experimentId') == before,
                  f'rollback={code} answered_by={pr3.get("experimentId") if isinstance(pr3, dict) else pr3} production={before}')
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


def forecast_scenario(mloop, budget, keep, expected, serve=False):
    base = tempfile.mkdtemp(prefix='lifecycle-forecast-')
    run = Run(mloop, 'forecast', base, budget, expected)
    rng = random.Random(20261103)
    a, new_a = make_series(0, 240, 'a', rng), None
    new_a = make_series(240, FORECAST_HORIZON, 'a', rng)
    b, new_b = make_series(240, 240, 'b', rng), None
    new_b = make_series(480, FORECAST_HORIZON, 'b', rng)
    header = ('value',)
    for name, rows in [('train_a', a), ('new_a', new_a), ('test_a', a + new_a), ('train_b', b), ('new_b', new_b), ('test_b', b + new_b)]:
        write_table(os.path.join(base, name + '.csv'), rows, header)
    print(f'== forecast  ({base})', flush=True)
    try:
        proj = os.path.join(base, 'proj')
        rc, init = run.json_of('init', 'proj', '--task', 'forecasting', '--label', 'value', '--json')
        run.check('init', rc == 0 and init and init.get('task') == 'forecasting', f'rc={rc}')

        rc, evs = run.events('train', os.path.join(base, 'train_a.csv'), '--label', 'value', *run.budget_args(), '--json', cwd=proj)
        result = next((e for e in evs if e.get('event') == 'result'), {})
        promo = next((e for e in evs if e.get('event') == 'promotion'), {})
        mae = (result.get('metrics') or {}).get('mae')
        # the swing is 10 and the noise 1: a model that learned the season misses by far less than the swing
        run.check('train', rc == 0 and result.get('experimentId') == 'exp-001' and promo.get('promoted') is True and mae is not None and mae < 6,
                  f'mae={mae} promoted={promo.get("promoted")}')

        rc, status = run.json_of('status', '--json', cwd=proj)
        models = (status or {}).get('models') or [{}]
        run.check('status', models[0].get('productionExperimentId') == 'exp-001', f'production={models[0].get("productionExperimentId")}')

        rc, fa = run.json_of('predict', '--json', cwd=proj)
        preds = (fa or {}).get('predictions') or []
        err_a = forecast_error(preds, new_a)
        run.check('forecast', rc == 0 and len(preds) == FORECAST_HORIZON and all(p.get('scoreLowerBound') is not None and p['scoreLowerBound'] <= p['score'] <= p['scoreUpperBound'] for p in preds)
                  and err_a is not None and err_a < 6, f'{len(preds)} steps, error against the values that followed={err_a}')

        rc, ev = run.json_of('evaluate', 'exp-001', os.path.join(base, 'test_a.csv'), '--json', cwd=proj)
        mae_a = ((ev or {}).get('testMetrics') or {}).get('mae')
        run.check('evaluate_in_regime', rc == 0 and mae_a is not None and mae_a < 3, f'test mae={mae_a} (the values that followed the series)')

        rc, drift = run.json_of('drift', '--json', cwd=proj)
        # a series has no feature columns to profile: drift says so, and exits cleanly
        run.check('drift_names_a_series_without_a_profile', rc == 0 and (drift or {}).get('status') == 'NoProfile', f'rc={rc} status={(drift or {}).get("status")}')

        rc, trig = run.json_of('trigger', 'check', '--json', cwd=proj)
        run.check('trigger', rc == 0 and trig is not None, f'rc={rc} trigger={json.dumps(trig)[:200] if trig else None}')

        rc, evs = run.events('train', os.path.join(base, 'train_b.csv'), '--label', 'value', *run.budget_args(), '--json', cwd=proj)
        result = next((e for e in evs if e.get('event') == 'result'), {})
        promo = next((e for e in evs if e.get('event') == 'promotion'), {})
        candidate = result.get('experimentId')
        run.check('retrain', rc == 0 and candidate == 'exp-002', f'{candidate} promoted={promo.get("promoted")} reason={(promo.get("keptInStagingReason") or "")[:90]}')

        # the same question put to both models: what comes next in the new regime
        rc, fc = run.json_of('predict', '--experiment', candidate or 'exp-002', '--json', cwd=proj)
        err_candidate = forecast_error((fc or {}).get('predictions') or [], new_b)
        rc, fp = run.json_of('predict', '--json', cwd=proj)
        err_production = forecast_error((fp or {}).get('predictions') or [], new_b)
        # production replays train_a: its forecast continues the old regime, so it misses the new one by about the shift
        run.check('retrained_forecasts_the_new_regime', err_candidate is not None and err_production is not None and err_candidate < 5 and err_candidate < err_production,
                  f'candidate error={err_candidate} production error={err_production} candidate answered={json.dumps(fc)[:200] if fc else None}')

        rc, ev = run.json_of('evaluate', candidate or 'exp-002', os.path.join(base, 'test_b.csv'), '--against', 'production', '--json', cwd=proj)
        verdict = (((ev or {}).get('comparison') or {}).get('paired') or {}).get('verdict')
        run.check('paired_evaluation', rc == 0 and verdict == 'CandidateBetter', f'rc={rc} verdict={verdict} comparison={json.dumps((ev or {}).get("comparison"))[:160]}')

        rc, pr = run.json_of('promote', candidate or 'exp-002', '--json', cwd=proj)
        rc2, hist = run.json_of('promote', '--history', '--json', cwd=proj)
        changes = (hist or {}).get('promotions') or []
        last = changes[-1] if changes else {}
        run.check('explicit_promote', (pr or {}).get('status') in ('promoted', 'already-production') and last.get('experimentId') == candidate,
                  f'status={(pr or {}).get("status")} history={len(changes)} last={last.get("experimentId")}/{last.get("gate")}')

        rc, rb = run.json_of('rollback', '--json', cwd=proj)
        run.check('rollback', (rb or {}).get('status') == 'rolled-back' and (rb or {}).get('experimentId') == 'exp-001', f'status={(rb or {}).get("status")} to={(rb or {}).get("experimentId")}')

        # the project says how a challenger is judged: against production, on the stretch it has to forecast
        config_path = os.path.join(proj, 'mloop.yaml')
        text = open(config_path).read()
        marker = '    description: '
        at = text.index(marker)
        eol = text.index('\n', at) + 1
        gate = f'    gate:\n      data: {os.path.join(base, "test_b.csv")}\n      no_worse_than_production: true\n'
        open(config_path, 'w').write(text[:eol] + gate + text[eol:])
        rc, evs = run.events('train', os.path.join(base, 'train_b.csv'), '--label', 'value', *run.budget_args(), '--json', cwd=proj)
        result = next((e for e in evs if e.get('event') == 'result'), {})
        promo = next((e for e in evs if e.get('event') == 'promotion'), {})
        rc2, hist = run.json_of('promote', '--history', '--json', cwd=proj)
        changes = (hist or {}).get('promotions') or []
        last = changes[-1] if changes else {}
        run.check('retrain_with_paired_gate', rc == 0 and promo.get('promoted') is True and last.get('experimentId') == result.get('experimentId') and last.get('gate') == 'passed',
                  f'{result.get("experimentId")} promoted={promo.get("promoted")} reason={promo.get("keptInStagingReason")} history last={last.get("experimentId")}/{last.get("gate")}')

        # the series a model forecasts after is kept beside it: the files it was trained from go away and the forecast stands
        for name in ('train_a', 'train_b'):
            os.rename(os.path.join(base, name + '.csv'), os.path.join(base, name + '.moved'))
        rc, fm = run.json_of('predict', '--json', cwd=proj)
        err_moved = forecast_error((fm or {}).get('predictions') or [], new_b)
        rc2, fc_moved = run.json_of('predict', '--experiment', 'exp-001', '--json', cwd=proj)
        err_moved_a = forecast_error((fc_moved or {}).get('predictions') or [], new_a)
        run.check('forecast_survives_the_training_files_going', rc == 0 and err_moved is not None and err_moved < 5 and err_moved_a is not None and err_moved_a < 6,
                  f'production error={err_moved} exp-001 error={err_moved_a}')

        # a series arrives most often with its dates beside it: the other columns are read and left alone
        import datetime
        day0 = datetime.date(2024, 1, 1)
        dated = lambda rows, t0: [[(day0 + datetime.timedelta(days=t0 + i)).isoformat(), r[0]] for i, r in enumerate(rows)]
        for name, rows, t0 in [('dated_a', a, 0), ('dated_test_a', a + new_a, 0)]:
            write_table(os.path.join(base, name + '.csv'), dated(rows, t0), ('date', 'value'))
        rc, out, err = run.cli('init', 'datedproj', '--task', 'forecasting', '--label', 'value')
        dproj = os.path.join(base, 'datedproj')
        rc, evs = run.events('train', os.path.join(base, 'dated_a.csv'), '--label', 'value', *run.budget_args(), '--json', cwd=dproj)
        result = next((e for e in evs if e.get('event') == 'result'), {})
        mae_d = (result.get('metrics') or {}).get('mae')
        run.check('dated_series_train', rc == 0 and mae_d is not None and mae_d < 3, f'mae={mae_d}')
        rc, fd = run.json_of('predict', '--json', cwd=dproj)
        err_d = forecast_error((fd or {}).get('predictions') or [], new_a)
        run.check('dated_series_forecast', rc == 0 and err_d is not None and err_d < 6, f'error against the values that followed={err_d}')
        rc, ev = run.json_of('evaluate', 'exp-001', os.path.join(base, 'dated_test_a.csv'), '--json', cwd=dproj)
        run.check('dated_series_evaluate', rc == 0 and ((ev or {}).get('testMetrics') or {}).get('mae') is not None, f'mae={((ev or {}).get("testMetrics") or {}).get("mae")}')

        # a missing point is a blank line in a file of one column; it is a gap, and neither a training nor an evaluation reads over it
        gap_path = os.path.join(base, 'gap.csv')
        values = [r[0] for r in a]
        values[100] = ''
        open(gap_path, 'w').write('value\n' + '\n'.join(map(str, values)) + '\n')
        rc, out, err = run.cli('init', 'gapproj', '--task', 'forecasting', '--label', 'value')
        rc, out, err = run.cli('train', gap_path, '--label', 'value', *run.budget_args(), cwd=os.path.join(base, 'gapproj'))
        run.check('gap_in_a_series_is_refused_by_training', rc != 0 and 'missing value' in (out + err), f'rc={rc}')
        rc, out, err = run.cli('evaluate', 'exp-001', gap_path, cwd=proj)
        run.check('gap_in_a_series_is_refused_by_evaluate', rc != 0 and 'missing value' in (out + err), f'rc={rc}')

        if serve:
            forecast_serve_steps(run, proj, new_a, new_b)
    except Exception as ex:
        run.check('scenario', False, f'{type(ex).__name__}: {ex}')
    finally:
        if keep:
            print(f'   kept {base}')
        else:
            shutil.rmtree(base, ignore_errors=True)
    return run.steps


ANOMALY_SPIKE = 40.0


def anomaly_scenario(mloop, budget, keep, expected):
    """A series watched for points that do not belong: rows in, a verdict per row out. The detector reads the file's order, so the
    walk's files are series (one column) with spikes put where the walk knows them."""
    base = tempfile.mkdtemp(prefix='lifecycle-anomaly-')
    run = Run(mloop, 'anomaly', base, budget, expected)
    rng = random.Random(20261103)

    def series(t0, n, spikes):
        rows = make_series(t0, n, 'a', rng)
        for at in spikes:
            rows[at][0] = round(rows[at][0] + ANOMALY_SPIKE, 4)
        return rows

    train_a, new_a = series(0, 300, [100, 200]), series(300, 200, [30, 70, 150])
    train_b = series(400, 300, [150])
    for name, rows in [('train_a', train_a), ('new_a', new_a), ('train_b', train_b)]:
        write_table(os.path.join(base, name + '.csv'), rows, ('value',))
    print(f'== anomaly  ({base})', flush=True)
    try:
        proj = os.path.join(base, 'proj')
        rc, init = run.json_of('init', 'proj', '--task', 'time-series-anomaly', '--label', 'value', '--json')
        run.check('init', rc == 0 and init and init.get('task') == 'time-series-anomaly', f'rc={rc}')

        rc, evs = run.events('train', os.path.join(base, 'train_a.csv'), '--label', 'value', *run.budget_args(), '--json', cwd=proj)
        result = next((e for e in evs if e.get('event') == 'result'), {})
        promo = next((e for e in evs if e.get('event') == 'promotion'), {})
        run.check('train', rc == 0 and result.get('experimentId') == 'exp-001' and promo.get('promoted') is True,
                  f'metrics={json.dumps(result.get("metrics"))[:160]} promoted={promo.get("promoted")}')

        rc, status = run.json_of('status', '--json', cwd=proj)
        models = (status or {}).get('models') or [{}]
        run.check('status', models[0].get('productionExperimentId') == 'exp-001', f'production={models[0].get("productionExperimentId")}')

        rc, pa = run.json_of('predict', os.path.join(base, 'new_a.csv'), '--log', '--json', cwd=proj)
        preds = (pa or {}).get('predictions') or []
        flagged = [i for i, p in enumerate(preds) if p.get('isAnomaly')]
        found = [at for at in (30, 70, 150) if any(abs(i - at) <= 2 for i in flagged)]
        early = [(i, preds[i].get('isAnomaly'), preds[i].get('anomalyScore')) for i in (0, 1, 30, 63, 64, 65)]
        run.check('predict_finds_the_spikes', rc == 0 and len(preds) == 200 and {70, 150} <= set(found) and len(flagged) <= 20,
                  f'{len(preds)} rows, flagged={flagged[:20]} found={found} rows (index, flag, score)={early}')
        notes = ' '.join((pa or {}).get('warnings') or [])
        # the detector reads a window of earlier points: the rows before the window carry no verdict, and the answer says so
        run.check('predict_says_which_rows_it_did_not_judge', all(preds[i].get('isAnomaly') is None for i in (0, 1, 30)) and preds[150].get('isAnomaly') is not None
                  and 'were not judged' in notes, f'rows 0/1/30 flag={[preds[i].get("isAnomaly") for i in (0, 1, 30)]} warnings={notes[:140]}')
        run.check('predict_log', all(p.get('predictionId') for p in preds) and len(preds) == 200, f'ids on {sum(1 for p in preds if p.get("predictionId"))} rows')

        rc, drift = run.json_of('drift', '--json', cwd=proj)
        run.check('drift_names_a_series_without_a_profile', rc == 0 and (drift or {}).get('status') == 'NoProfile', f'rc={rc} status={(drift or {}).get("status")}')

        rc, evs = run.events('train', os.path.join(base, 'train_b.csv'), '--label', 'value', *run.budget_args(), '--json', cwd=proj)
        result = next((e for e in evs if e.get('event') == 'result'), {})
        candidate = result.get('experimentId')
        run.check('retrain', rc == 0 and candidate == 'exp-002', f'{candidate} promoted={next((e for e in evs if e.get("event") == "promotion"), {}).get("promoted")}')

        rc, pr = run.json_of('promote', candidate or 'exp-002', '--json', '--force', cwd=proj)
        run.check('explicit_promote', (pr or {}).get('status') in ('promoted', 'already-production'), f'rc={rc} status={(pr or {}).get("status")}')
        rc, rb = run.json_of('rollback', '--json', cwd=proj)
        run.check('rollback', (rb or {}).get('status') == 'rolled-back' and (rb or {}).get('experimentId') == 'exp-001', f'status={(rb or {}).get("status")} to={(rb or {}).get("experimentId")}')
    except Exception as ex:
        run.check('scenario', False, f'{type(ex).__name__}: {ex}')
    finally:
        if keep:
            print(f'   kept {base}')
        else:
            shutil.rmtree(base, ignore_errors=True)
    return run.steps


def unlabeled_scenario(mloop, task, budget, keep, expected):
    """A task without a label: two groups of points, rows in, a group or an outlier verdict out. What moves is the inputs, so drift
    and the retraining trigger are the walk's questions (a feedback has no actual value to carry)."""
    base = tempfile.mkdtemp(prefix=f'lifecycle-{task}-')
    run = Run(mloop, task, base, budget, expected)
    rng = random.Random(20261103)

    def points(n, shift=0.0, outliers=()):
        rows = []
        for _ in range(n):
            c = rng.choice([0, 1])
            rows.append([round(rng.gauss(5 * c, 1) + shift, 3), round(rng.gauss(5 * c, 1), 3)])
        for at in outliers:
            rows[at] = [40.0, -40.0]
        return rows

    for name, rows in [('train_a', points(300)), ('new_a', points(400, 0, (10, 50))), ('new_b', points(400, 6.0))]:
        write_table(os.path.join(base, name + '.csv'), rows, ('x', 'y'))
    print(f'== {task}  ({base})', flush=True)
    try:
        proj = os.path.join(base, 'proj')
        rc, init = run.json_of('init', 'proj', '--task', TASK_FLAG[task], '--json')
        run.check('init', rc == 0 and init and init.get('task') == TASK_FLAG[task], f'rc={rc}')
        rc, evs = run.events('train', os.path.join(base, 'train_a.csv'), *run.budget_args(), '--json', cwd=proj)
        result = next((e for e in evs if e.get('event') == 'result'), {})
        promo = next((e for e in evs if e.get('event') == 'promotion'), {})
        run.check('train', rc == 0 and result.get('experimentId') == 'exp-001' and promo.get('promoted') is True, f'metrics={json.dumps(result.get("metrics"))[:140]}')
        rc, pa = run.json_of('predict', os.path.join(base, 'new_a.csv'), '--log', '--json', cwd=proj)
        preds = (pa or {}).get('predictions') or []
        run.check('predict_log', rc == 0 and len(preds) == 400 and all(p.get('predictionId') for p in preds), f'{len(preds)} rows')
        if task == 'outliers':
            far = [i for i in (10, 50) if preds and preds[i].get('isAnomaly')]
            run.check('predict_finds_the_far_points', far == [10, 50], f'flagged among the two planted={far}')
        else:
            groups = {p.get('clusterId') for p in preds}
            run.check('predict_names_the_groups', groups == {0, 1}, f'clusterIds={sorted(g for g in groups if g is not None)}')
        rc, drift = run.json_of('drift', '--json', cwd=proj)
        run.check('drift_control', rc == 0 and ((drift or {}).get('drift') or {}).get('level') == 'None', f'level={((drift or {}).get("drift") or {}).get("level")}')
        rc, pb = run.json_of('predict', os.path.join(base, 'new_b.csv'), '--log', '--json', cwd=proj)
        rc, drift = run.json_of('drift', '--json', cwd=proj)
        run.check('drift_shifted', rc == 0 and ((drift or {}).get('drift') or {}).get('level') == 'Significant', f'level={((drift or {}).get("drift") or {}).get("level")}')
        rc, trig = run.json_of('trigger', 'check', '--json', cwd=proj)
        run.check('trigger', rc == 0 and (trig or {}).get('shouldRetrain') is True, f'shouldRetrain={(trig or {}).get("shouldRetrain")}')
    except Exception as ex:
        run.check('scenario', False, f'{type(ex).__name__}: {ex}')
    finally:
        if keep:
            print(f'   kept {base}')
        else:
            shutil.rmtree(base, ignore_errors=True)
    return run.steps


def grouped_scenario(mloop, task, budget, keep, expected):
    """Rows that belong to a group - the documents of a query, the ratings of a user - scored per row. A ranking's relevance is judged
    only relevant or not (0 and 1), as it is in most data; a recommendation's rating runs 1 to 5. The second regime reverses what is liked."""
    base = tempfile.mkdtemp(prefix=f'lifecycle-{task}-')
    run = Run(mloop, task, base, budget, expected)
    rng = random.Random(20261103)
    if task == 'ranking':
        header, label = ('query_id', 'f1', 'f2', 'relevance'), 'relevance'

        def table(n_queries, q0, flip=False):
            rows = []
            for q in range(q0, q0 + n_queries):
                for _ in range(10):
                    f1, f2 = rng.random(), rng.random()
                    s = (-f1 * 2 + f2 + 1.0) if flip else (f1 * 2 + f2 - 1.0)
                    rows.append([q, round(f1, 3), round(f2, 3), 1 if s + rng.gauss(0, 0.15) > 0.6 else 0])
            return rows
        files = {'train_a': table(80, 0), 'new_a': table(30, 100), 'train_b': table(80, 200, True), 'test_b': table(30, 300, True)}
    else:
        header, label = ('user_id', 'item_id', 'rating'), 'rating'

        def table(n, flip=False):
            rows = []
            for _ in range(n):
                u, i = rng.randrange(40), rng.randrange(40)
                like = 3.5 if (u % 4) == (i % 4) else 1.5
                rows.append([f'u{u}', f'i{i}', round(min(5, max(1, (5 - like + 1 if flip else like) + rng.gauss(0, 0.3))), 2)])
            return rows
        files = {'train_a': table(3000), 'new_a': table(300), 'train_b': table(3000, True), 'test_b': table(1000, True)}
    for name, rows in files.items():
        write_table(os.path.join(base, name + '.csv'), rows, header)
    print(f'== {task}  ({base})', flush=True)
    try:
        proj = os.path.join(base, 'proj')
        rc, init = run.json_of('init', 'proj', '--task', TASK_FLAG[task], '--label', label, '--json')
        run.check('init', rc == 0 and init and init.get('task') == TASK_FLAG[task], f'rc={rc}')
        rc, evs = run.events('train', os.path.join(base, 'train_a.csv'), '--label', label, *run.budget_args(), '--json', cwd=proj)
        result = next((e for e in evs if e.get('event') == 'result'), {})
        promo = next((e for e in evs if e.get('event') == 'promotion'), {})
        run.check('train', rc == 0 and result.get('experimentId') == 'exp-001' and promo.get('promoted') is True, f'metrics={json.dumps(result.get("metrics"))[:120]}')
        rc, pa = run.json_of('predict', os.path.join(base, 'new_a.csv'), '--log', '--json', cwd=proj)
        preds = (pa or {}).get('predictions') or []
        run.check('predict_log', rc == 0 and len(preds) == len(files['new_a']) and all(p.get('predictionId') and p.get('score') is not None for p in preds), f'{len(preds)} rows')
        rc, ev = run.json_of('evaluate', 'exp-001', os.path.join(base, 'new_a.csv'), '--json', cwd=proj)
        run.check('evaluate', rc == 0 and bool((ev or {}).get('testMetrics')), f'rc={rc}')
        rc, drift = run.json_of('drift', '--json', cwd=proj)
        if task == 'ranking':
            # a query id is a name for a group, not an input: new queries bring new ids and must not read as drift
            lvl = ((drift or {}).get('drift') or {}).get('level')
            cols = [f.get('column') for f in ((drift or {}).get('drift') or {}).get('features', [])]
            run.check('drift_control', rc == 0 and lvl == 'None' and 'query_id' not in cols, f'level={lvl} columns={cols}')
        else:
            run.check('drift_names_a_model_without_a_profile', rc == 0 and (drift or {}).get('status') == 'NoProfile', f'status={(drift or {}).get("status")}')
        rc, evs = run.events('train', os.path.join(base, 'train_b.csv'), '--label', label, *run.budget_args(), '--json', cwd=proj)
        result = next((e for e in evs if e.get('event') == 'result'), {})
        promo = next((e for e in evs if e.get('event') == 'promotion'), {})
        candidate = result.get('experimentId')
        run.check('retrain', rc == 0 and candidate == 'exp-002', f'{candidate} promoted={promo.get("promoted")}')
        if promo.get('promoted') is True:
            # the new regime can score higher on its own rows than the old model did on its: promoted by its recorded score, which is the
            # rule working as written; the slot goes back so the paired step has a production to compare against
            rc, rb = run.json_of('rollback', '--json', cwd=proj)
            run.check('rollback_of_auto_promotion', (rb or {}).get('status') == 'rolled-back' and (rb or {}).get('experimentId') == 'exp-001', f'status={(rb or {}).get("status")}')
        rc, ev = run.json_of('evaluate', candidate or 'exp-002', os.path.join(base, 'test_b.csv'), '--against', 'production', '--json', cwd=proj)
        comparison = (ev or {}).get('comparison') or {}
        if task == 'recommendation':
            verdict = (comparison.get('paired') or {}).get('verdict')
            run.check('paired_evaluation', rc == 0 and verdict == 'CandidateBetter', f'verdict={verdict}')
        else:
            # a ranking is judged per query, which has no row-wise measure: the comparison says so rather than guessing
            run.check('paired_evaluation_says_what_it_cannot_pair', rc == 0 and comparison.get('paired') is None and bool(comparison.get('notPaired')), f'notPaired={comparison.get("notPaired")}')
        rc, pr = run.json_of('promote', candidate or 'exp-002', '--json', '--force', cwd=proj)
        run.check('explicit_promote', (pr or {}).get('status') in ('promoted', 'already-production'), f'status={(pr or {}).get("status")}')
        rc, rb = run.json_of('rollback', '--json', cwd=proj)
        run.check('rollback', (rb or {}).get('status') == 'rolled-back' and (rb or {}).get('experimentId') == 'exp-001', f'status={(rb or {}).get("status")}')
    except Exception as ex:
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
            if task == 'text':
                # the libtorch runtime is installed on demand; the scenario trains a text model, so it is part of the walk
                installed = subprocess.run(shlex.split(args.mloop) + ['runtime', 'install', 'torch'], capture_output=True, text=True)
                steps.append({'step': 'text/runtime_install', 'status': 'pass' if installed.returncode == 0 else 'FAIL',
                              'detail': (installed.stdout + installed.stderr)[-200:].strip()})
            if task in ('ranking', 'recommendation'):
                steps += grouped_scenario(args.mloop, task, args.time, args.keep, expected)
                continue
            if task in ('clustering', 'outliers'):
                steps += unlabeled_scenario(args.mloop, task, args.time, args.keep, expected)
                continue
            if task == 'anomaly':
                steps += anomaly_scenario(args.mloop, args.time, args.keep, expected)
                continue
            if task == 'forecast':
                steps += forecast_scenario(args.mloop, args.time, args.keep, expected, not args.no_serve)
                continue
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
