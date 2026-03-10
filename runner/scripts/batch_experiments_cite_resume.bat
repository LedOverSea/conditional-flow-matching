echo 2.OT-CFMLitModule...
python "src/train.py" "experiment=cite" "model=otcfm" "ckpt_path=D:/desktop/code/conditional-flow-matching/runner/logs/train/runs/2026-03-09_23-24-48/checkpoints/epoch_1099.ckpt"

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