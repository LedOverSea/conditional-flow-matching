@echo off
echo 0. Null ...
python "src/train.py" "experiment=multi" "model=cfm"  

echo 1. Exact ...
python "src/train.py" "experiment=multi" "model=cfm"  "model.ot_sampler=exact"

echo 2. Sinkhorn ...
python "src/train.py" "experiment=multi" "model=cfm"  "model.ot_sampler=sinkhorn"

echo 3. Unbalanced ...
python "src/train.py" "experiment=multi" "model=cfm"  "model.ot_sampler=unbalanced"

echo 4. Partial ...
python "src/train.py" "experiment=multi" "model=cfm"  "model.ot_sampler=partial"

echo experiments complete!
pause