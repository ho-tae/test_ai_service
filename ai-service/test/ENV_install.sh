pip install wheel
pip install numpy==1.19.4
pip install torch==1.6.0
pip install torchvision==0.7.0
pip install scikit-build
pip install scikit-image
pip install mmcv-full==1.2.0
cd inference_module
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
cd ..
export PYTHONPATH=$`pwd`/inference_module
