#!/usr/bin/env python3
"""Measure the `error_rise` retraining signal on public regression tables: how often does it fire when nothing changed, when the
model got worse, and does weighting repair a feedback sample drawn with unequal probabilities?

The signal (MLoop.Ops, `FeedbackBasedTrigger`): the feedback's weighted RMSE against the RMSE the production experiment recorded on the
rows it held out at training. It is met when the feedback is worth at least 30 rows (Kish's effective size) and the 95 % lower bound of the
feedback RMSE is at least `k` times that baseline. This script reproduces that rule (the same arithmetic, here in Python) over real models:

  per table and seed   rows are shuffled with the seed, the first half trains a model (`mloop train`, the time limit given), the second half is
                       a pool of rows the model never saw; `mloop predict` answers the pool; error = actual - answer.
  M1  no change        draw n rows from the pool, 500 times, for n in 30 / 100 / 300: how often is the rule met, for each k
  M2  got worse        the actual values get gaussian noise (sd = sqrt(c^2 - 1) x baseline RMSE) for c in 1.5 / 2 / 3: the same count
  M3  weighted sample  review probability proportional to the rank of the answer's size, Poisson sampling at about n = 100, weights 1/pi:
                       how far the weighted and the unweighted mean squared error are from the pool's own (relative bias, mean over draws)

  python error-rise-measure.py --mloop "dotnet tools/MLoop.CLI/bin/Release/net10.0/mloop.dll" --source <openml-benchmarks dir> --out <dir>
         [--tables airfoil_self_noise,houses] [--extra wine_quality] [--seeds 3] [--time 20] [--draws 500]

Without --tables it takes every TabArena regression table of 1,500 to 50,000 rows with 3 to 10 numeric features (the pre-registered set);
--extra adds tables that are reported apart. Standard library only. The pool of a large table is capped at 3,000 rows. The table files stay where
they are read: nothing of them is copied beside this script, and the output holds numbers only.
"""
import argparse, csv, json, math, os, random, shlex, shutil, subprocess, sys

Z = 1.959964            # MLoop.Ops RetrainingDefaults.AccuracyConfidenceZ
MIN_N = 30              # MLoop.Ops RetrainingDefaults.MinimumAccuracyFeedback
KS = (1.0, 1.25, 1.5, 2.0)
NS = (30, 100, 300)
CS = (1.5, 2.0, 3.0)


def fires(errs, weights, baseline, k):
    """The rule: effective size >= MIN_N and the 95 % lower bound of the weighted RMSE >= k x baseline."""
    sw = sum(weights)
    if sw <= 0:
        return False
    n_eff = sw * sw / sum(w * w for w in weights)
    if n_eff < MIN_N:
        return False
    sq = [e * e for e in errs]
    mse = sum(w * s for w, s in zip(weights, sq)) / sw
    var = sum(w * (s - mse) ** 2 for w, s in zip(weights, sq)) / sw
    lower = mse - Z * math.sqrt(var / n_eff)
    return math.sqrt(max(0.0, lower)) >= k * baseline


def read_manifest(source):
    with open(os.path.join(source, 'MANIFEST.csv'), newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def pre_registered(rows):
    out = []
    for r in rows:
        if r['task_type'] != 'regression' or 'TabArena' not in r['suites'] or not r['target']:
            continue
        n, num = float(r['rows']), float(r['NumberOfNumericFeatures'])
        if 1500 <= n <= 50000 and 3 <= num <= 10:
            out.append(r)
    return sorted(out, key=lambda r: r['name'])


def cli(mloop, *args, cwd):
    p = subprocess.run(mloop + list(args), cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def cell(mloop, row, source, seed, work, budget):
    """Train on half of a shuffled table and answer the other half. Returns (baseline rmse, errors) or None."""
    path = os.path.join(source, 'data', row['file'])
    with open(path, newline='', encoding='utf-8-sig') as f:
        rd = csv.reader(f)
        header = next(rd)
        rows = list(rd)
    label = row['target']
    if label not in header:
        return None, f'label {label!r} not in header'
    li = header.index(label)
    rng = random.Random(seed)
    rng.shuffle(rows)
    half = len(rows) // 2
    train, pool = rows[:half], rows[half:half + 3000]
    name = f"{row['name']}-s{seed}"
    proj = os.path.join(work, name)
    shutil.rmtree(proj, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    # `init` takes a project name, not a path: it makes the directory under the current one
    rc, out, err = cli(mloop, 'init', name, '--task', 'regression', '--label', label, cwd=work)
    if rc != 0:
        return None, f'init rc={rc} {(out + err)[-200:]}'
    tr = os.path.join(proj, 'train_half.csv')
    with open(tr, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(train)
    rc, out, err = cli(mloop, 'train', tr, '--label', label, '--time', str(budget), '--json', cwd=proj)
    result = None
    for line in out.splitlines():
        if line.startswith('{') and '"event":"result"' in line:
            result = json.loads(line)
    if rc != 0 or not result or 'rmse' not in (result.get('metrics') or {}):
        return None, f'train rc={rc} {(out + err)[-200:]}'
    baseline = result['metrics']['rmse']
    nl = os.path.join(proj, 'pool_nolabel.csv')
    with open(nl, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header[:li] + header[li + 1:])
        w.writerows(r[:li] + r[li + 1:] for r in pool)
    outp = os.path.join(proj, 'answers.csv')
    rc, out, err = cli(mloop, 'predict', nl, '-o', outp, cwd=proj)
    if rc != 0 or not os.path.exists(outp):
        return None, f'predict rc={rc} {(out + err)[-200:]}'
    with open(outp, newline='', encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        answers = [float(a['Score']) for a in rd]
    if len(answers) != len(pool):
        return None, f'answers {len(answers)} != pool {len(pool)}'
    actual = [float(r[li]) for r in pool]
    errs = [a - p for a, p in zip(actual, answers)]
    return {'baseline': baseline, 'errs': errs, 'answers': answers}, None


def analyse(c, draws, rng):
    errs, baseline, answers = c['errs'], c['baseline'], c['answers']
    pool_rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
    res = {'baseline': baseline, 'pool_rmse': pool_rmse, 'pool_n': len(errs), 'm1': {}, 'm2': {}, 'm3': {}}
    for n in NS:
        if n > len(errs):
            continue
        hit = {k: 0 for k in KS}
        for _ in range(draws):
            s = [errs[i] for i in rng.sample(range(len(errs)), n)]
            for k in KS:
                hit[k] += fires(s, [1.0] * n, baseline, k)
        res['m1'][n] = {k: hit[k] / draws for k in KS}
    for cc in CS:
        sd = math.sqrt(cc * cc - 1) * baseline
        noisy = [e + rng.gauss(0, sd) for e in errs]
        res['m2'][cc] = {'realised_ratio': math.sqrt(sum(e * e for e in noisy) / len(noisy)) / baseline}
        for n in NS:
            if n > len(errs):
                continue
            hit = {k: 0 for k in KS}
            for _ in range(draws):
                s = [noisy[i] for i in rng.sample(range(len(noisy)), n)]
                for k in KS:
                    hit[k] += fires(s, [1.0] * n, baseline, k)
            res['m2'][cc][n] = {k: hit[k] / draws for k in KS}
    # M3: review probability proportional to the rank of |answer|; about 100 rows per draw
    order = sorted(range(len(answers)), key=lambda i: abs(answers[i]))
    rank = [0] * len(answers)
    for r, i in enumerate(order, 1):
        rank[i] = r
    total_rank = sum(rank)
    target = min(100, len(errs) // 2)
    pi = [min(1.0, target * rank[i] / total_rank) for i in range(len(errs))]
    truth = sum(e * e for e in errs) / len(errs)
    wb, ub, sizes = [], [], []
    for _ in range(draws):
        idx = [i for i in range(len(errs)) if rng.random() < pi[i]]
        if len(idx) < 5:
            continue
        sizes.append(len(idx))
        sq = [errs[i] ** 2 for i in idx]
        w = [1.0 / pi[i] for i in idx]
        wb.append(sum(a * b for a, b in zip(w, sq)) / sum(w) / truth - 1)
        ub.append(sum(sq) / len(sq) / truth - 1)
    res['m3'] = {'weighted_bias': sum(wb) / len(wb), 'unweighted_bias': sum(ub) / len(ub), 'mean_rows': sum(sizes) / len(sizes)}
    return res


def median(xs):
    xs = sorted(xs)
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def summarise(cells, label):
    lines = [f'### {label} — {len(cells)} cells (table x seed)', '']
    lines.append('| M1 no change: mean fire share (worst cell) | ' + ' | '.join(f'k={k}' for k in KS) + ' |')
    lines.append('|---|' + '---|' * len(KS))
    for n in NS:
        cs = [c['m1'][n] for c in cells if n in c['m1']]
        lines.append(f'| n={n} | ' + ' | '.join(f'{sum(x[k] for x in cs) / len(cs):.3f} ({max(x[k] for x in cs):.2f})' for k in KS) + ' |')
    lines.append('')
    for cc in CS:
        real = median([c['m2'][cc]['realised_ratio'] for c in cells])
        lines.append(f'| M2 got worse c={cc} (realised ratio, median {real:.2f}): mean fire share | ' + ' | '.join(f'k={k}' for k in KS) + ' |')
        lines.append('|---|' + '---|' * len(KS))
        for n in NS:
            cs = [c['m2'][cc][n] for c in cells if n in c['m2'][cc]]
            lines.append(f'| n={n} | ' + ' | '.join(f'{sum(x[k] for x in cs) / len(cs):.3f}' for k in KS) + ' |')
        lines.append('')
    wb = [c['m3']['weighted_bias'] for c in cells]
    ub = [c['m3']['unweighted_bias'] for c in cells]
    lines.append(f'M3 weighted sample (median over cells of the mean relative bias of the mean squared error): weighted {median(wb):+.3f} · unweighted {median(ub):+.3f}'
                 f' · worst cell weighted {max(wb, key=abs):+.3f} · rows per draw ≈ {median([c["m3"]["mean_rows"] for c in cells]):.0f}')
    ratios = [c['pool_rmse'] / c['baseline'] for c in cells]
    lines.append(f'Runner check (no change): pool RMSE / baseline RMSE per cell — median {median(ratios):.2f}, min {min(ratios):.2f}, max {max(ratios):.2f}')
    lines.append('')
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mloop', required=True)
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tables')
    ap.add_argument('--extra', default='')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--time', type=int, default=20)
    ap.add_argument('--draws', type=int, default=500)
    a = ap.parse_args()
    mloop = shlex.split(a.mloop)
    rows = read_manifest(a.source)
    by_name = {}
    for r in rows:
        if r['task_type'] == 'regression' and r['target']:
            by_name.setdefault(r['name'], r)
    main_rows = [by_name[n] for n in a.tables.split(',')] if a.tables else pre_registered(rows)
    extra_rows = [by_name[n] for n in a.extra.split(',') if n]
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(20261120)
    groups = {'pre-registered tables': main_rows, 'informational tables': extra_rows}
    store = {g: [] for g in groups}
    skipped = []
    for g, trs in groups.items():
        for row in trs:
            for seed in range(1, a.seeds + 1):
                c, why = cell(mloop, row, a.source, seed, os.path.join(a.out, 'work'), a.time)
                if c is None:
                    skipped.append((row['name'], seed, why))
                    print(f"skip {row['name']} seed {seed}: {why}", file=sys.stderr)
                    continue
                r = analyse(c, a.draws, rng)
                r['table'], r['seed'] = row['name'], seed
                store[g].append(r)
                print(f"{row['name']} s{seed}: baseline {r['baseline']:.4g} pool {r['pool_rmse']:.4g} "
                      f"M1 n100 k1.5 {r['m1'].get(100, {}).get(1.5, float('nan')):.2f} M2 c2 n100 k1.5 {r['m2'][2.0].get(100, {}).get(1.5, float('nan')):.2f}", flush=True)
    with open(os.path.join(a.out, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump({g: [{k: v for k, v in c.items()} for c in cs] for g, cs in store.items()}, f, indent=1, default=str)
    lines = ['# error_rise measurement', '']
    for g, cs in store.items():
        if cs:
            lines += summarise(cs, g)
    if skipped:
        lines += ['Skipped cells: ' + '; '.join(f'{t} s{s}: {w}' for t, s, w in skipped), '']
    text = '\n'.join(lines)
    with open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
