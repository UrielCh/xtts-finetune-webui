@echo off
conda create -n py310 python=3.10
call conda activate py310

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt

python xtts_demo.py