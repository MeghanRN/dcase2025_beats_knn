#!/usr/bin/env bash
# ------------------------------------------------------------
# download_task2_data.sh
# Downloads all DCASE-2025-Task-2 archives (dev + eval)
# into data/dcase2025t2/{dev_data,eval_data}/raw/…
# Skips anything that already exists.
# ------------------------------------------------------------
set -euo pipefail

fetch () {
  local url=$1 zip=$2
  echo "  ↪ $zip"
  wget -q --show-progress --continue -O "$zip" "$url"
  unzip -q "$zip"
  rm "$zip"
}

echo "== Development data =="
mkdir -p data/dcase2025t2/dev_data/raw
pushd  data/dcase2025t2/dev_data/raw >/dev/null
for m in ToyCar ToyTrain bearing fan gearbox slider valve; do
  [[ -d $m ]] && { echo "✓ $m exists – skip"; continue; }
  fetch "https://zenodo.org/records/15097779/files/dev_${m}.zip" "dev_${m}.zip"
done
popd >/dev/null

echo "== Additional-train & Eval-test data =="
mkdir -p data/dcase2025t2/eval_data/raw
pushd  data/dcase2025t2/eval_data/raw >/dev/null
for m in AutoTrash HomeCamera ToyPet ToyRCCar BandSealer Polisher ScrewFeeder CoffeeGrinder; do
  [[ -d ${m}/train ]] || \
    fetch "https://zenodo.org/records/15392814/files/eval_data_${m}_train.zip" "eval_data_${m}_train.zip"
  [[ -d ${m}/test  ]] || \
    fetch "https://zenodo.org/records/15519362/files/eval_data_${m}_test.zip"  "eval_data_${m}_test.zip"
done
popd >/dev/null

echo "All Task-2 data present ✅"