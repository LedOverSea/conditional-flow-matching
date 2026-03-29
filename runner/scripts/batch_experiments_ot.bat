@echo off
echo 0. Null ...
echo 0.1 eb_phate
python "src/train.py" "experiment=eb_phate" "model=cfm"  

echo 0.2 eb_pca
python "src/train.py" "experiment=eb_pca" "model=cfm"  

echo 0.3 mutli
python "src/train.py" "experiment=mutli" "model=cfm"  

echo 0.4 cite
python "src/train.py" "experiment=cite" "model=cfm"  

echo 1. Exact ...
echo 1.1 eb_phate
python "src/train.py" "experiment=eb_phate" "model=cfm"  "model.ot_sampler=exact"

echo 1.2 eb_pca
python "src/train.py" "experiment=eb_pca" "model=cfm"  "model.ot_sampler=exact"

echo 1.3 mutli
python "src/train.py" "experiment=mutli" "model=cfm"  "model.ot_sampler=exact"

echo 1.4 cite
python "src/train.py" "experiment=cite" "model=cfm"  "model.ot_sampler=exact"

echo 2. Sinkhorn ...
echo 2.1 eb_phate
python "src/train.py" "experiment=eb_phate" "model=cfm"  "model.ot_sampler=sinkhorn"

echo 2.2 eb_pca
python "src/train.py" "experiment=eb_pca" "model=cfm"  "model.ot_sampler=sinkhorn"

echo 2.3 mutli
python "src/train.py" "experiment=mutli" "model=cfm"  "model.ot_sampler=sinkhorn"

echo 2.4 cite
python "src/train.py" "experiment=cite" "model=cfm"  "model.ot_sampler=sinkhorn"

echo 3. Unbalanced ...
echo 3.1 eb_phate
python "src/train.py" "experiment=eb_phate" "model=cfm"  "model.ot_sampler=unbalanced"

echo 3.2 eb_pca
python "src/train.py" "experiment=eb_pca" "model=cfm"  "model.ot_sampler=unbalanced"

echo 3.3 mutli
python "src/train.py" "experiment=mutli" "model=cfm"  "model.ot_sampler=unbalanced"

echo 3.4 cite
python "src/train.py" "experiment=cite" "model=cfm"  "model.ot_sampler=unbalanced"

echo 4. Partial ...
echo 4.1 eb_phate
python "src/train.py" "experiment=eb_phate" "model=cfm"  "model.ot_sampler=partial"

echo 4.2 eb_pca
python "src/train.py" "experiment=eb_pca" "model=cfm"  "model.ot_sampler=partial"

echo 4.3 mutli
python "src/train.py" "experiment=mutli" "model=cfm"  "model.ot_sampler=partial"

echo 4.4 cite  
python "src/train.py" "experiment=cite" "model=cfm"  "model.ot_sampler=partial"

echo experiments complete!
pause