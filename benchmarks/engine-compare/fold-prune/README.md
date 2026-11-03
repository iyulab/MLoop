# fold-prune

Reads the per-fold metrics of each trial (`trial_log[].validation.foldMetrics` in the result rows of an `Engine compare` run
with `systems=mloop`) and asks whether the first fold of a trial ranks it the way the mean over all folds does, and what a
"drop a trial whose first fold is below the q-th quantile of earlier first folds" rule would save.

    python3 -I foldrank.py results-*.jsonl      # rank correlation per (table, seed) run
    python3 -I prune_rules.py results-*.jsonl   # saved share of trial time and what the rule would have dropped, per quantile

Tables with a single holdout (15,000 rows and more) have no folds and are skipped.
