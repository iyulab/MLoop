#!/usr/bin/env python3
"""Walk a text task's model through its operating life with the real CLI and say, step by step, whether each stage behaved.

The table tasks have `lifecycle-check.py`. The deep-learning text tasks (NER, question answering, sentence similarity) answer a
tag sequence, a text and a number, so what a table model's life measures needs a reading of its own: how feedback judges the
answer, what the drift profile holds, what a paired comparison can say. This walks one project per task, on generated English text
(no corpus, nothing downloaded; standard library only):

  1  init + train             a first model learns the task and, being the first, is promoted by itself
  2  status / history         the project names it as production; the history records the gate as passed
  3  predict --log            one answer per row, each logged with an id; `export onnx` says why it is refused
  4  drift profile            training wrote the text columns and no other (a question-answering table's answer position is not
                              an input — profiled, it read as the largest drift there is on every window)
  5  drift                    with too few logged rows it says so; with 300+ rows drawn from the training distribution it reads
                              them and finds no significant drift, and `trigger check` does not ask for retraining on it
  6  feedback + trigger       a class or a text answer is matched (half of the answers wrong → the accuracy condition is met);
                              a value answer that states no interval (similarity) has no accuracy: it says so and does not ask
                              for retraining — "the same value" is not a measure of a number
  7  retrain, compare         a second model; `compare` ties them; `evaluate --against production` says there is no row to pair
  8  promote, rollback        the slot moves to it and back; the history records both
  9  serve                    /predict answers what the CLI answers and logs with ids; a promotion and a rollback made while the
                              model sits in the server's cache reach the next answer
 10  --job                    an async `POST /train` finishes and, asked not to promote, says why the model stays in staging

  python text-task-lifecycle.py --mloop "dotnet tools/MLoop.CLI/bin/Release/net10.0/mloop.dll" [--tasks ner,qa,sim] [--work DIR] [--job]

Local only: it needs the libtorch runtime (`mloop runtime install torch`). Each task trains for a fixed number of epochs (the time
limit does not shorten it): about 6 to 11 minutes a model, two models a task (three with --job). Tasks are independent, so start
one process per task to walk them side by side. Exit status is 1 when a check fails. A JSON summary is written to <work>/<task>/summary.json.
"""
import argparse, csv, importlib.util, itertools, json, os, random, shlex, shutil, signal, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('lifecycle_check', os.path.join(HERE, 'lifecycle-check.py'))
_lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lc)
http, free_port = _lc.http, _lc.free_port

TASKS = {   # task flag, label column, columns a prediction row carries
    'ner': ('ner', 'Label', ['Sentence']),
    'qa': ('question-answering', 'answer', ['context', 'question']),
    'sim': ('sentence-similarity', 'score', ['sentence1', 'sentence2']),
}

# ---- generated data ----------------------------------------------------------------------------------------------

PEOPLE = ['Alice', 'Bob', 'Carol', 'David', 'Emma', 'Frank', 'Grace', 'Henry', 'Irene', 'Jack', 'Karen', 'Leo']
CITIES = ['Paris', 'London', 'Berlin', 'Madrid', 'Rome', 'Vienna', 'Oslo', 'Lisbon', 'Dublin', 'Prague']
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
TIMES = ['7 pm', '9 am', 'noon', '6 pm', '8 am']
COMPANIES = ['Acme', 'Globex', 'Initech', 'Umbrella', 'Hooli', 'Vandelay', 'Stark', 'Wayne', 'Wonka', 'Tyrell']
FILLER = ['The office is busy.', 'It opened last year.', 'Lunch is served at noon.', 'Parking is limited.']


def ner_rows(rng, n):
    rows, seen = [], set()
    while len(rows) < n:
        p, c, d, t = rng.choice(PEOPLE), rng.choice(CITIES), rng.choice(DAYS), rng.choice(TIMES)
        time_tags = 'B-TIME' if ' ' not in t else 'B-TIME I-TIME'
        row = [(f'{p} lives in {c}', 'B-PER O O B-LOC'), (f'We meet {d} at {t}', f'O O B-DATE O {time_tags}'),
               (f'{p} flew to {c} on {d}', 'B-PER O O B-LOC O B-DATE'), (f'The train from {c} arrives at {t}', f'O O O B-LOC O O {time_tags}')][rng.randint(0, 3)]
        if row[0] not in seen:
            seen.add(row[0]); rows.append(row)
    return [{'Sentence': s, 'Label': t} for s, t in rows]


def qa_rows(rng, n):
    combos = list(itertools.product(PEOPLE, COMPANIES, CITIES))
    rng.shuffle(combos)
    rows = []
    for p, c, t in combos[:n]:
        ctx = f'{p} works at {c} in {t}. {rng.choice(FILLER)}'
        q, a = (f'Where does {p} work?', c) if rng.random() < .5 else (f'Which city is {c} in?', t)
        rows.append({'context': ctx, 'question': q, 'answer': a, 'answer_start': ctx.index(a)})
    return rows


def sim_rows(rng, n):
    pairs = [('the cat sat on the mat', 'a cat is sitting on a mat'), ('he plays the guitar every night', 'every night he plays guitar'),
             ('the market opens at nine', "the market opens at nine o'clock"), ('she reads a book in the garden', 'in the garden she is reading a book'),
             ('the engineer fixed the machine', 'the machine was repaired by the engineer'), ('rain is expected tomorrow', 'tomorrow it will probably rain'),
             ('the children play football', 'kids are playing soccer'), ('we ate dinner at home', 'dinner was at our house'),
             ('the bank closes early on friday', 'on friday the bank shuts early'), ('a dog runs in the park', 'the dog is running through the park')]
    unrelated = ['stock prices fell sharply', 'the volcano erupted overnight', 'my printer is out of paper', 'she won the chess tournament',
                 'the bridge will be repainted', 'bananas are rich in potassium', 'the meeting moved to room four', 'a new library opened downtown']
    rows = []
    for _ in range(n):
        a, b = rng.choice(pairs)
        if rng.random() < 2 / 3:
            x, y, s = (a, b, 5.0) if rng.random() < .5 else (b, a, 5.0)
        else:
            u = rng.choice(unrelated)
            x, y, s = (a, u, 0.0) if rng.random() < .5 else (u, b, 0.0)
        rows.append({'sentence1': x, 'sentence2': y, 'score': s})
    return rows


GENERATORS = {'ner': ner_rows, 'qa': qa_rows, 'sim': sim_rows}


def write_csv(path, rows, columns):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows([[r[c] for c in columns] for r in rows])


# ---- the walk ----------------------------------------------------------------------------------------------------

def first_json(text):
    starts = [i for i in (text.find('{'), text.find('[')) if i >= 0]
    return json.loads(text[min(starts):]) if starts else None


class Walk:
    def __init__(self, task, mloop, work):
        self.task, self.mloop, self.work = task, mloop, work
        self.flag, self.label, self.columns = TASKS[task]
        self.proj = os.path.join(work, f'{task}-proj')
        self.checks = []

    def cli(self, *args):
        p = subprocess.run(self.mloop + list(args), cwd=self.proj if os.path.isdir(self.proj) else self.work, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr

    def json(self, *args):
        rc, out, err = self.cli(*args)
        try:
            return rc, first_json(out), out + err
        except ValueError:
            return rc, None, out + err

    def check(self, name, ok, detail=''):
        self.checks.append({'check': name, 'ok': bool(ok), 'detail': str(detail)[:300]})
        print(f"{'PASS' if ok else 'FAIL'}  {self.task:3s} {name}  ({str(detail)[:200]})", flush=True)

    def answers(self, predictions):
        return [p.get('predictedLabel', p.get('score')) for p in predictions]


def walk(task, mloop, work, seed, job):
    w = Walk(task, mloop, work)
    os.makedirs(work, exist_ok=True)
    data = os.path.join(work, 'data')
    os.makedirs(data, exist_ok=True)
    gen = GENERATORS[task]
    allcols = w.columns + [w.label] + (['answer_start'] if task == 'qa' else [])
    rows = gen(random.Random(seed), 360 if task == 'qa' else 300 + 50)
    train, test = rows[:300], rows[300:]
    train_f, test_f, pred_f, traffic_f = (os.path.join(data, f'{n}.csv') for n in ('train', 'test', 'predict', 'traffic'))
    write_csv(train_f, train, allcols)
    write_csv(test_f, test, allcols)
    write_csv(pred_f, test[:20], w.columns)
    # Rows the model will meet, drawn from the same distribution as its training rows (a fresh draw, not a copy of the 20 above):
    # a window of repeated rows is not independent, and a drift reading of it says nothing about the model.
    traffic = gen(random.Random(seed + 1), 340)
    write_csv(traffic_f, traffic, w.columns)

    rc, out, err = w.cli('init', f'{task}-proj', '--task', w.flag)
    w.check('init', rc == 0, '' if rc == 0 else (out + err)[-200:])
    if rc != 0:
        return w
    t0 = time.time()
    rc, out, err = w.cli('train', train_f, '--label', w.label)
    w.check('train (first model)', rc == 0, f'{round(time.time() - t0)} s')
    if rc != 0:
        return w

    rc, status, _ = w.json('status', '--json')
    model = (status or {}).get('models', [{}])[0] if status else {}
    w.check('status names exp-001 as production', rc == 0 and model.get('hasProduction') is True and model.get('productionExperimentId') == 'exp-001', model.get('productionExperimentId'))
    rc, hist, _ = w.json('promote', '--history', '--json')
    first = ((hist or {}).get('promotions') or [{}])[0]
    w.check('history records the first promotion with the gate passed', first.get('action') == 'auto-promote' and first.get('gate') == 'passed', first)

    rc, pr, text = w.json('predict', pred_f, '--json', '--log')
    preds = (pr or {}).get('predictions') or []
    w.check('predict --log: one answer with an id per row', rc == 0 and len(preds) == 20 and all(p.get('predictionId') for p in preds), f'{len(preds)} answers')
    cli_answers = w.answers(preds)
    rc, out, err = w.cli('export', 'onnx', 'exp-001')
    w.check('export onnx is refused and names the task', rc != 0 and w.flag in (out + err), (out + err).strip()[-140:])

    # 4. the profile training wrote
    profile_path = os.path.join(w.proj, 'models', 'default', 'production', 'drift-profile.json')
    profile = json.load(open(profile_path, encoding='utf-8')) if os.path.exists(profile_path) else {}
    w.check('drift profile holds text columns and no other', profile.get('features') == [] and len(profile.get('texts') or []) == len(w.columns),
            f"features={[f.get('column') for f in profile.get('features', [])]} texts={[t.get('column') for t in profile.get('texts') or []]}")

    # 5. drift: too few rows, then enough
    rc, d, _ = w.json('drift', '--json')
    w.check('drift with 20 logged rows says it needs more', (d or {}).get('status') == 'TooFewRows', (d or {}).get('message', '')[:100])
    rc, out, err = w.cli('predict', traffic_f, '--log', '--json')
    w.check('predict --log of 340 fresh rows', rc == 0, (out + err)[-80:] if rc else '')
    rc, d, _ = w.json('drift', '--json')
    report = (d or {}).get('drift') or {}
    columns = [f.get('column') for f in report.get('features', [])]
    w.check('drift over 300+ in-distribution rows reads them and finds no significant drift',
            (d or {}).get('status') == 'Measured' and report.get('level') != 'Significant', f"{(d or {}).get('status')} {report.get('level')} {columns}")
    w.check('drift compares text descriptors only', bool(columns) and all(c.endswith('(length)') or c.endswith('(unseen words)') for c in columns), columns)
    rc, tr, _ = w.json('trigger', 'check', '--json')
    drift_cond = next((c for c in (tr or {}).get('conditions', []) if c.get('name') == 'input_drift'), {})
    w.check('trigger: input_drift is not met on in-distribution traffic', drift_cond.get('isMet') is False, drift_cond.get('details', '')[:140])

    # 6. feedback
    rc, logs, _ = w.json('logs', '--json', '--limit', '400')
    entries = (logs or [])[:60]
    fb = []
    for i, e in enumerate(entries):
        out_value = e.get('output')
        if task == 'sim':
            actual = round(float(out_value) + (0.2 if i % 2 else -0.2), 3)         # all within 0.2 of the answer: a good model
        elif task == 'qa':
            actual = out_value if i % 2 == 0 else 'something else'                  # half of the answers wrong
        else:
            actual = out_value if i % 2 == 0 else ' '.join(['O'] * (len(out_value.split()) - 1) + ['B-X'])
        fb.append((e['id'], actual))
    fb_f = os.path.join(data, 'feedback.csv')
    with open(fb_f, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['prediction_id', 'actual'])
        wr.writerows(fb)
    rc, out, err = w.cli('feedback', 'add', '--from-file', fb_f, '--id-column', 'prediction_id', '--actual-column', 'actual')
    w.check('feedback add --from-file records 60 rows', rc == 0 and 'recorded for 60 of 60' in out.lower(), out.strip()[-100:])
    rc, fm, _ = w.json('feedback', 'metrics', '--json')
    rc, tr, _ = w.json('trigger', 'check', '--json')
    acc = next((c for c in (tr or {}).get('conditions', []) if c.get('name') == 'accuracy_threshold'), {})
    if task == 'sim':
        w.check('a value answer without an interval has no accuracy', (fm or {}).get('accuracy') is None, (fm or {}).get('accuracy'))
        w.check('trigger: accuracy does not apply and does not ask for retraining', acc.get('isMet') is False and 'does not apply' in acc.get('details', ''), acc.get('details', '')[:160])
    else:
        a = (fm or {}).get('accuracy')
        w.check('a class or text answer is matched (half wrong → about 50 %)', a is not None and 0.35 <= a <= 0.65, a)
        w.check('trigger: accuracy condition is met on an established drop', acc.get('isMet') is True, acc.get('details', '')[:160])

    # 7. a second model
    t0 = time.time()
    rc, out, err = w.cli('train', train_f, '--label', w.label)
    w.check('train (second model)', rc == 0, f'{round(time.time() - t0)} s')
    if rc == 0:
        rc, cmp_, _ = w.json('compare', 'exp-001', 'exp-002', '--json')
        w.check('compare ranks both', rc == 0 and len((cmp_ or {}).get('ranking') or []) == 2, [r.get('id') for r in (cmp_ or {}).get('ranking') or []])
        rc, ev, text = w.json('evaluate', 'exp-002', test_f, '--against', 'production', '--json')
        not_paired = ((ev or {}).get('comparison') or {}).get('notPaired') or ''
        w.check('evaluate --against production says there is no row to pair', rc == 0 and 'no per-row measure' in not_paired, not_paired[:100])
        profile2 = os.path.join(w.proj, 'models', 'default', 'staging', 'exp-002', 'drift-profile.json')
        p2 = json.load(open(profile2, encoding='utf-8')) if os.path.exists(profile2) else {}
        w.check('the second model\'s drift profile holds text columns only', p2.get('features') == [] and bool(p2.get('texts')), [f.get('column') for f in p2.get('features', [])])
        rc, pro, _ = w.json('promote', 'exp-002', '--json')
        w.check('promote exp-002', rc == 0 and (pro or {}).get('status') == 'promoted', (pro or {}).get('status'))
        rc, rb, _ = w.json('rollback', '--json')
        w.check('rollback returns to exp-001', rc == 0 and (rb or {}).get('experimentId') == 'exp-001', (rb or {}).get('experimentId'))
        rc, hist, _ = w.json('promote', '--history', '--json')
        actions = [h.get('action') for h in (hist or {}).get('promotions') or []]
        w.check('history: auto-promote, promote, rollback', actions == ['auto-promote', 'promote', 'rollback'], actions)
        serve_steps(w, pred_f, cli_answers, job, train_f)
    return w


def serve_steps(w, pred_f, cli_answers, job, train_f):
    rc, token, _ = w.cli('token', '--role', 'admin', '-q')
    token = token.strip()
    port = free_port()
    url = f'http://127.0.0.1:{port}'
    server = subprocess.Popen(w.mloop + ['serve', '--port', str(port), '--host', '127.0.0.1'], cwd=w.proj,
                              env={**os.environ, 'Serve__RateLimit__PermitsPerMinute': '0'},
                              stdout=open(os.path.join(w.work, 'serve.log'), 'ab'), stderr=subprocess.STDOUT, start_new_session=True)
    try:
        up = False
        for _ in range(90):
            if http('GET', url + '/health', timeout=2)[0] == 200:
                up = True
                break
            time.sleep(1)
        w.check('serve is up', up and bool(token), f'port={port}')
        if not up:
            return
        rows = list(csv.DictReader(open(pred_f, newline='', encoding='utf-8')))
        code, r = http('POST', url + '/predict?name=default', token, rows)
        served = r.get('predictions') if isinstance(r, dict) else None
        w.check('POST /predict answers what the CLI answered, each with an id',
                code == 200 and served is not None and w.answers(served) == cli_answers and all(p.get('predictionId') for p in served),
                f'code={code} n={len(served or [])}')
        before = r.get('experimentId') if isinstance(r, dict) else None
        code, _ = http('POST', url + '/promote', token, {'name': 'default', 'experimentId': 'exp-002'})
        code2, r2 = http('POST', url + '/predict?name=default', token, rows[:3])
        w.check('a promotion made while the model is cached reaches the next answer', code == 200 and isinstance(r2, dict) and r2.get('experimentId') == 'exp-002',
                f"promote={code} answered_by={r2.get('experimentId') if isinstance(r2, dict) else r2}")
        code, _ = http('POST', url + '/rollback', token, {'name': 'default'})
        code2, r3 = http('POST', url + '/predict?name=default', token, rows[:3])
        w.check('a rollback reaches the next answer', code == 200 and isinstance(r3, dict) and r3.get('experimentId') == before,
                f"rollback={code} answered_by={r3.get('experimentId') if isinstance(r3, dict) else r3} expected={before}")
        if job:
            os.makedirs(os.path.join(w.proj, 'datasets'), exist_ok=True)
            shutil.copy(train_f, os.path.join(w.proj, 'datasets', 'job_train.csv'))
            code, j = http('POST', url + '/train', token, {'dataFile': 'datasets/job_train.csv', 'name': 'default', 'labelColumn': w.label, 'task': w.flag, 'promote': False})
            jid = (j or {}).get('jobId') if isinstance(j, dict) else None
            final, t0 = {}, time.time()
            while jid and time.time() - t0 < 3000:
                _, final = http('GET', f'{url}/jobs/{jid}', token)
                if isinstance(final, dict) and str(final.get('status', '')).lower() in ('completed', 'failed'):
                    break
                time.sleep(5)
            w.check('POST /train job completes and says why it stays in staging',
                    code == 202 and str(final.get('status', '')).lower() == 'completed' and final.get('promoted') is False and bool(final.get('keptInStagingReason')),
                    f"accepted={code} status={final.get('status')} reason={str(final.get('keptInStagingReason'))[:80]}")
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tasks', default='ner,qa,sim')
    ap.add_argument('--work', default=None)
    ap.add_argument('--mloop', default=os.environ.get('MLOOP', 'mloop'), help='the mloop command, e.g. "dotnet …/mloop.dll" (env MLOOP)')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--job', action='store_true', help='also run an async POST /train job (one more model per task)')
    a = ap.parse_args()
    work = a.work or tempfile.mkdtemp(prefix='text-task-lifecycle-')
    failed = 0
    for task in [t.strip() for t in a.tasks.split(',') if t.strip()]:
        if task not in TASKS:
            sys.exit(f'unknown task {task!r}: one of {", ".join(TASKS)}')
        w = walk(task, shlex.split(a.mloop), os.path.join(work, task), a.seed, a.job)
        failed += sum(not c['ok'] for c in w.checks)
        with open(os.path.join(work, task, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(w.checks, f, indent=2)
    print(f'\n{failed} failed — work: {work}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
