# Image Classficiation
Mixed-precision training for ResNet50 v1.5 modified from [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets).

In this example, we use ActNN with Average Quantization by manually constructing the model with the memory-saving layers.

## Requirements
- Put the ImageNet dataset to `~/imagenet`
- Install required packages
```bash
pip install matplotlib tqdm
```

## Train ResNet50 v1.5 by Actnn_AQ on ImageNet (Full Precision)
```
./dist-train 1 0 127.0.0.1 1 resnet50 \
   "-c quantize --ca=True --actnn-level L3 -bit 1.5 --aq-bit 1.0"\
   tmp ~/imagenet 256
```
