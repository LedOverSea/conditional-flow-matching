@echo off
echo 1.CFMLitModule...
python "src/1214.py" "model._target_=src.models.cfm_module.CFMLitModule"

echo 2.RectifiedFlowLitModule...
python "src/1214.py" "model._target_=src.models.cfm_module.RectifiedFlowLitModule" 

echo 3.ActionMatchingLitModule...
python "src/1214.py" "model._target_=src.models.cfm_module.ActionMatchingLitModule"

echo 4.VariancePreservingCFM...
python "src/1214.py" "model._target_=src.models.cfm_module.VariancePreservingCFMLitModule"

echo 5.SBCFMLitModule...
python "src/1214.py" "model._target_=src.models.cfm_module.SBCFMLitModule"

echo 6.SF2MLitModule...
python "src/1214.py" "model=sf2m" 

echo mission complete！
pause