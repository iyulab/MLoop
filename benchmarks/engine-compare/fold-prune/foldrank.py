"""Does the first fold's score rank a trial the way the mean over all folds will?
usage: python3 -I foldrank.py results-*.jsonl
Per (dataset, seed) run: Spearman rank correlation between fold-1 score and final mean of the trials that have fold metrics,
and the precision of a prune rule 'drop a trial whose fold-1 score is worse than the Q-th quantile of earlier trials' fold-1 scores'
(how many of the dropped trials were really in the worse half / worse-than-best-so-far by final mean).
"""
import json, sys, statistics as st
PRIMARY = {'binary': ('auc', +1), 'multiclass': ('log_loss', -1), 'regression': ('rmse', -1)}

def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]: j += 1
        for k in range(i, j + 1): r[order[k]] = (i + j) / 2
        i = j + 1
    return r

def spearman(a, b):
    if len(a) < 4: return None
    ra, rb = rank(a), rank(b); ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else None

rows = []
for path in sys.argv[1:]:
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        if r.get('system') != 'mloop': continue
        metric, sign = PRIMARY[r['task']]
        pts = []
        for t in r.get('trial_log') or []:
            v = t.get('validation') or {}
            fm = v.get('foldMetrics')
            mean = (t.get('metrics') or {}).get(metric)
            if not fm or mean is None or any(metric not in f for f in fm): continue
            pts.append((sign * fm[0][metric], sign * mean, [sign * f[metric] for f in fm], t['runtimeSeconds'], t['trialNumber']))
        rows.append((r['dataset'], r['seed'], pts))

if not any(p for _, _, p in rows):
    print('no trial carries foldMetrics'); sys.exit(1)
print('dataset                     seed  trials  rho(f1,mean)  rank-1-fold kept-best?  saved-time-if-prune')
rhos = []
for ds, seed, pts in rows:
    if len(pts) < 4: print(f'{ds:26s} {seed:4d}  {len(pts):5d}  (too few)'); continue
    rho = spearman([p[0] for p in pts], [p[1] for p in pts]); rhos.append(rho)
    # prune rule: after the first 4 trials, drop a trial whose fold-1 is below the median of the fold-1 scores of the previous trials
    kept_best = True; saved = 0.0; total = sum(p[3] for p in pts)
    best_final = max(p[1] for p in pts)
    for i, p in enumerate(pts):
        if i >= 4 and p[0] < st.median(q[0] for q in pts[:i]):
            # a pruned trial's remaining folds are not run: saved share = (folds-1)/folds of its time
            folds = len(p[2]); saved += p[3] * (folds - 1) / folds
            if p[1] == best_final: kept_best = False
    print(f'{ds:26s} {seed:4d}  {len(pts):5d}  {rho if rho is None else round(rho, 2):>12}  {str(kept_best):>22}  {100 * saved / total:5.1f} %')
print('median rho', round(st.median(r for r in rhos if r is not None), 2), 'over', len(rhos), 'runs')
