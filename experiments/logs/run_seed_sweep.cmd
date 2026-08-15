@echo off
cd /d D:\research\jet_tagging
D:\research\jet_tagging\.venv\Scripts\python.exe -u -m scripts.train_arcface --seed 123 --epochs 50 --no-chunk --output_dir experiments\arcface_seed_123 --checkpoint_dir experiments\arcface_seed_123 1> experiments\arcface_seed_sweep.log 2> experiments\arcface_seed_sweep.log.err
D:\research\jet_tagging\.venv\Scripts\python.exe -u -m scripts.train_arcface --seed 7 --epochs 50 --no-chunk --output_dir experiments\arcface_seed_7 --checkpoint_dir experiments\arcface_seed_7 1>> experiments\arcface_seed_sweep.log 2>> experiments\arcface_seed_sweep.log.err