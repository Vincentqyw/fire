conda install faiss-gpu

cd lib/asmk 
python3 setup.py build_ext --inplace
rm -r build
cd ../../

## optional 
pip3 install -r lib/how/requirements.txt


## run model

python evaluate_single_image.py eval_fire.yml

