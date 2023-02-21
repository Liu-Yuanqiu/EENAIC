# EENAIC
This repository contains the reference code for the paper "End-to-End Non-Autoregressive Image Captioning" (ICASSP 2023).

# Requirements
- Python 3.7
- Pytorch 1.12
- Torchvision 0.13
- timm 0.6.11
- numpy
- tqdm

# Preparation
## 1. Data preparation
The necessary files in training and evaluation are saved in `mscoco` folder, which is organized as follows:
```python
mscoco/
|--feature/
  |--coco2014/
    |--train2014/
    |--value2014/
    |--test2014/
|--misc/
|--sent/
|--txt/
```
Download annotation files from [GoogleDrive](https://drive.google.com/file/d/1muydp9MVCUY4-hoKTsJvk4dZoycnrzbj/view?usp=sharing) and the images from [MSCOCO 2014](https://cocodataset.org/#download). Put the images into `mscoco/feature/coco2014`.

## 2. Pre-trained model preparation
Download pre-trained Backbone model (Swin-Transformer) from [GoogleDrive](https://drive.google.com/file/d/17NI_cHE8wKEB8-pbtlKVDcxY6dgPFCkD/view?usp=sharing) and save it in the root directory.

# Training
```python
bash experiments/EENAIC/train.sh
```

# Evaluation
You can download the pre-trained model from [GoogleDrive](https://drive.google.com/file/d/19y8ZcNIMdqrqaw5K6lOgb90wxxrY690b/view?usp=sharing) and put it into `experiments/EENAIC/snapshot/`.
```python
bash experiments/EENAIC/eval.sh
```
| BLEU-1      | BLEU-2      | BLEU-3      | BLEU-4      | METEOR      | ROUGE-L     | CIDEr       |     
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| 79.7        | 63.2        | 48.5        | 36.9        | 27.9        | 58.0        | 122.6       |

# Acknowledgements
Thanks the original [PureT](https://github.com/232525/PureT) and [JDAI-CV/image-captioning](https://github.com/JDAI-CV/image-captioning).
