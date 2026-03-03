@echo off

echo 1.eb_phate...
python "src/train.py" "experiment=eb_phate" "model=am"

echo 2.eb_pca...
python "src/train.py" "experiment=eb_pca" "model=am"

echo 3.multi...
python "src/train.py" "experiment=multi" "model=am"

echo 4.cite...
python "src/train.py" "experiment=cite" "model=am"

echo experiments complete!
pause