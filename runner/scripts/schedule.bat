@echo off
echo 执行第一个命令...
python src/train.py trainer.max_epochs=5 logger=csv

echo 执行第二个命令...
python src/train.py trainer.max_epochs=10 logger=csv

echo 执行完成！
pause