@echo off
echo 1.CFMLitModule...
python "src/train_eb_pca.py" "model._target_=src.models.cfm_module.CFMLitModule"

echo 2.RectifiedFlowLitModule...
python "src/train_eb_pca.py" "model._target_=src.models.cfm_module.RectifiedFlowLitModule"

echo 3.ActionMatchingLitModule...
python "src/train_eb_pca.py" "model._target_=src.models.cfm_module.ActionMatchingLitModule" "model.net._target_=src.models.components.simple_mlp.EnergyVelocityNet"

echo 4.VariancePreservingCFM...
python "src/train_eb_pca.py" "model._target_=src.models.cfm_module.VariancePreservingCFM"

echo 5.SBCFMLitModule...
python "src/train_eb_pca.py" "model._target_=src.models.cfm_module.SBCFMLitModule"

echo 6.SF2MLitModule...
python "src/train_eb_pca.py" "model=sf2m" 

echo experiments complete!
pause