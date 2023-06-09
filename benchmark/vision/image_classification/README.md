# Image Classficiation
Training for ResNet50 v1.5 modified from [actnn/image_classification](https://github.com/ucbrise/actnn/tree/main/image_classification).

In this example, we use ActNN_AQ by manually constructing the model with the memory-saving layers.

## Requirements
- Put the ImageNet dataset to `~/imagenet`
- Install required packages
```bash
pip install matplotlib tqdm
```

## Train ResNet50 v1.5 by Actnn_AQ on ImageNet 
- bit : target average precision
- aq-bit : utilize AQ 0.x-bit instead of 1-bit (e.g. AQ 0.5-bit: group average of 4 elements and 2-bit quantization, 2/4-bit)
```
chmod +x dist-train

./dist-train 1 0 127.0.0.1 8 29500 resnet50 \
   "-c quantize --ca=True --actnn-level L3 --bit 1.5 --aq-bit 1.0"\
   tmp ~/imagenet 256
```
