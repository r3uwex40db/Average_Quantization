# Benchmark Memory Usage and Training Speed on Torchvision Models

Benchmark Memory Usage and Training Speed modified from [actnn/mem_speed_benchmark](https://github.com/ucbrise/actnn/tree/main/mem_speed_benchmark).

## Prepare dataset
Put the ImageNet dataset to `~/imagenet`

## Benchmark Memory Usage
```
DEBUG_MEM=True python3 train.py ~/imagenet --arch ARCH -b BATCH_SIZE --alg ALGORITHM --bit BIT --aq-bit AQ-BIT
```
The choices for ARCH are {resnet50, resnet152, wide_resnet101_2, densenet201}  

The choices for ALGORITHM are {exact, actnn-L0, actnn-L1, actnn-L2, actnn-L3, actnn-L4, actnn-L5}

For example, the memory required to ActNN with AQ 0.5-bit when target average precision is 1.25-bit can be acheived by
```
DEBUG_MEM=True python3 train.py ~/imagenet --arch resnet50 -b 128 --alg actnn-L3 --bit 1.25 --aq-bit 0.5
```

## Benchmark Training Speed
```
DEBUG_SPEED=True python3 train.py ~/imagenet --arch ARCH -b BATCH_SIZE --alg ALGORITHM
```
The choices for ARCH are {resnet50, resnet152, wide_resnet101_2, densenet201}  

The choices for ALGORITHM are {exact, actnn-L0, actnn-L1, actnn-L2, actnn-L3, actnn-L4, actnn-L5}  

For example, the training speed by ActNN with AQ 0.5-bit when target average precision is 1.25-bit can be acheived by
```
DEBUG_SPEED=True python3 train.py ~/imagenet --arch resnet50 -b 128 --alg actnn-L3 --bit 1.25 --aq-bit 0.5
```

