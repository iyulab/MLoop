"""Pre-registered prune rules evaluated on recorded trial logs (fold-1 score vs previous trials' fold-1 scores).
Rules: quantile q of previous fold-1 scores (after warm-up 4 trials), pruned trial saves (folds-1)/folds of its runtime.
Reports per rule: mean saved share of trial time (all folded runs / the four lagging tables), runs where the final best was pruned,
pruned trials that ended within the top-3 of the run by final mean.
"""
import json, sys, statistics as st
PRIMARY = {'binary': ('auc', +1), 'multiclass': ('log_loss', -1), 'regression': ('rmse', -1)}
LAG = {'students_dropout_and_academic_success', 'qsar-biodeg', 'airfoil_self_noise', 'wine_quality'}
def quant(xs, q):
    xs = sorted(xs); i = q * (len(xs) - 1); lo = int(i); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)
runs = []
for path in sys.argv[1:]:
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        if r.get('system') != 'mloop': continue
        metric, sign = PRIMARY[r['task']]; pts = []
        for t in r.get('trial_log') or []:
            fm = (t.get('validation') or {}).get('foldMetrics'); mean = (t.get('metrics') or {}).get(metric)
            if not fm or mean is None or any(metric not in f for f in fm): continue
            pts.append((sign * fm[0][metric], sign * mean, len(fm), t['runtimeSeconds']))
        if len(pts) >= 8: runs.append((r['dataset'], r['seed'], pts))
print('runs with folded trials:', len(runs))
for q in (0.5, 0.35, 0.25, 0.15):
  for margin in (0.0,):
    saved_all, saved_lag, lost_best, lost_top3, pruned_n, n = [], [], 0, 0, 0, 0
    for ds, seed, pts in runs:
        final_sorted = sorted(range(len(pts)), key=lambda i: -pts[i][1]); top3 = set(final_sorted[:3]); best = final_sorted[0]
        total = sum(p[3] for p in pts); sv = 0.0; lb = False
        for i, p in enumerate(pts):
            if i >= 4 and p[0] < quant([x[0] for x in pts[:i]], q):
                sv += p[3] * (p[2] - 1) / p[2]; pruned_n += 1
                if i == best: lb = True
                if i in top3: lost_top3 += 1
        n += len(pts); lost_best += lb
        (saved_lag if ds in LAG else saved_all).append(sv / total)
    allv = saved_all + saved_lag
    print(f'q={q:4}: saved mean {100*st.mean(allv):4.1f} % (lagging four {100*st.mean(saved_lag):4.1f} %), median {100*st.median(allv):4.1f} %; best pruned in {lost_best}/{len(runs)} runs; top-3 trials pruned {lost_top3}; pruned {pruned_n}/{n} trials')
