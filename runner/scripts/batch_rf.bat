@echo off

echo 1.eb_phate...
python "src/train.py" "experiment=eb_phate" "model=rf"

echo 2.eb_pca...
python "src/train.py" "experiment=eb_pca" "model=rf"

echo 3.multi...
python "src/train.py" "experiment=multi" "model=rf"

echo 4.cite...
python "src/train.py" "experiment=cite" "model=rf"

echo experiments complete!
pause