#!/usr/bin/env bash
# Ask whether the served model should be retrained, and start the retraining workflow when it should.
#
# Run it from cron on the machine that serves the model: its project folder holds the prediction log and the
# feedback the conditions are measured from.
#
#   0 6 * * *  /srv/churn/watch-production.sh
#
# `mloop trigger check` always exits 0 - the answer is `shouldRetrain` in its JSON.
set -euo pipefail
cd "$(dirname "$0")"

mloop trigger check --json > trigger.json

if [ "$(jq -r '.shouldRetrain' trigger.json)" = "true" ]; then
  jq -r '.conditions[] | select(.isMet) | "\(.name): \(.details)"' trigger.json
  gh workflow run retrain-and-promote.yml
fi
