# Learning Super-Features for Image Retrieval

Fork from [https://github.com/naver/FIRe](https://github.com/naver/FIRe), removing dependences of several submodules via adding these moudules to [lib](https://github.com/Vincentqyw/fire/tree/main/lib) with path setting. 


This repo is integrated to [Hierarchical-Localization](https://github.com/Vincentqyw/Hierarchical-Localization) pipline by adding [hloc/extractors/fire.py](https://github.com/Vincentqyw/Hierarchical-Localization/blob/master/hloc/extractors/fire.py) interface.

This repository contains the code for running FIRe model presented in our [ICLR'22 paper](https://openreview.net/pdf?id=wogsFPHwftY):

```bibtex
@inproceedings{superfeatures,
  title={{Learning Super-Features for Image Retrieval}},
  author={{Weinzaepfel, Philippe and Lucas, Thomas and Larlus, Diane and Kalantidis, Yannis}},
  booktitle={{ICLR}},
  year={2022}
}
```

## License

The code is distributed under the CC BY-NC-SA 4.0 License. See [LICENSE](LICENSE) for more information.
It is based on code from [HOW](https://github.com/gtolias/how), [cirtorch](https://github.com/filipradenovic/cnnimageretrieval/) and [ASMK](https://github.com/jenicek/asmk) that are released under their own license, the MIT license.

## Preparation


1. install ASMK

```
pip3 install pyaml numpy faiss-gpu 
cd lib/asmk
python3 setup.py build_ext --inplace
rm -r build
```

2. install dependencies by running:

```
pip3 install -r requirements.txt
```

3. data/experiments folders

All data will be stored under a folder ``fire_data`` that will be created when running the code; similarly, results and models from all experiments will be stored under folder ``fire_experiments``

## Evaluating our ICLR'22 FIRe model

To evaluate on ROxford/RParis our model trained on SfM-120k, simply run
```
python evaluate.py eval_fire.yml
```

With the released model and the parameters found in ``eval_fire.yml``, we obtain 90.3 on the validation set, 82.6 and 62.2 on ROxford medium and hard respectively, 85.2 and 70.0 on RParis medium and hard respectively.


## Training a FIRe model 

Simply run
```
python train.py train_fire.yml -e train_fire
```
All training outputs will be saved to ``fire_experiments/train_fire``.

To evaluate the trained model that was saved in ``fire_experiments/train_fire``, simply run:
```
python evaluate.py eval_fire.yml -e train_fire -ml train_fire
```

## Pretrained models

For reproducibility, we provide the following model weights for the architecture we use in the paper (ResNet50 without the last block + LIT):
* Model pre-trained on ImageNet-1K (with Cross-Entropy, the pre-trained model we use for training FIRe) [(link)](http://download.europe.naverlabs.com/ComputerVision/FIRe/pretraining/fire_imagenet.pth) or [fire_imagenet.pth](https://github.com/Vincentqyw/fire/blob/main/model/fire_imagenet.pth)
* Model trained on SfM-120k trained with FIRe [(link)](http://download.europe.naverlabs.com/ComputerVision/FIRe/official/fire.pth) or [fire_SfM_120k.pth](https://github.com/Vincentqyw/fire/blob/main/model/fire_SfM_120k.pth)

They will be automatically downloaded when running the training / testing script.
