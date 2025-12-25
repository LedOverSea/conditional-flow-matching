@echo off

echo 3.ActionMatchingLitModule...
python "src/train_eb_phate.py" "model._target_=src.models.cfm_module.ActionMatchingLitModule" "model.net._target_=src.models.components.simple_mlp.EnergyVelocityNet"

echo 4.VariancePreservingCFM...
python "src/train_eb_phate.py" "model._target_=src.models.cfm_module.VariancePreservingCFM"

echo 3.ActionMatchingLitModule...
python "src/train_eb_pca.py" "model._target_=src.models.cfm_module.ActionMatchingLitModule" "model.net._target_=src.models.components.simple_mlp.EnergyVelocityNet"

echo 4.VariancePreservingCFM...
python "src/train_eb_pca.py" "model._target_=src.models.cfm_module.VariancePreservingCFM"

echo experiments complete!
pause