@echo off
echo 1.I-CFMLitModule...
python "src/train.py" "experiment=cite" "model=cfm"

echo 2.OT-CFMLitModule...
python "src/train.py" "experiment=cite" "model=otcfm"

echo 3.RectifiedFlowLitModule...
python "src/train.py" "experiment=cite" "model=rf" 

echo 4.ActionMatchingLitModule...
python "src/train.py" "experiment=cite" "model=am"

echo 5.VariancePreservingCFM...
python "src/train.py" "experiment=cite" "model=vp"

echo 6.SBCFMLitModule...
python "src/train.py" "experiment=cite" "model=sbcfm"

echo 7.SF2MLitModule...
python "src/train.py" "experiment=cite" "model=sf2m"

echo experiments complete!
pause