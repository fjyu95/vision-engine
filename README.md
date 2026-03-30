# vision-engine
A high-performance vision inference engine with pipeline orchestration, multi-GPU scheduling, and modular algorithm integration.

conda create -n vision-engine python=3.10
conda activate vision-engine
pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128
# torch260+cu124+flash273
# torch280+cu128+flash283
pip install -r requirements.txt