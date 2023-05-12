# Average Quantization: High-Quality 0.x-bit Activation to Reduce Memory Usage for Training

This repository is the official implementation of https://openreview.net/forum?id=BG7H1XsMG0.

+ We prove that approximating activations with their average values minimizes the gradient variance from actual activations.
+  We propose Average Quantization which can generate high-quality sub-1b activations.
+  By adopting the proposed sub-1b precision in sensitivity-based activation compression training, we can allocate more bits to activations with high sensitivity than with previous algorithms, thus improving the accuracy-memory saving trade-off.

## Abstract

As the size of deep learning models increases, their performance improves according to scaling laws. However, training such large models requires a significant amount of memory and hence cannot be realized without a considerable number of expensive GPUs. Recently proposed activation compression techniques alleviate memory requirements by compressing activations with different compression rates based on their sensitivity to minimize accuracy degradation. However, when we deeply compress activations with the target precision close to 1 bit, a large portion of activations would be represented in 1 bit regardless of their sensitivity since sub-1b precision is not supported in those methods, severely degrading training accuracy. To address this issue, we propose to represent a group of activations with a single approximate value, effectively producing sub-1b activations. We prove that the boundary function for the gradient difference between the actual and the approximate activation is minimized when the activations are approximated using their average values. Based on this observation, we propose Average Quantization which provides high-quality sub-1b activations by replacing a group of activations with a single average value. By assigning sub-1b precision to activations with low sensitivity, we can allocate more bits to activations with higher sensitivity, which can result in a better trade-off between accuracy and memory saving. In experiments, the proposed Average Quantization successfully trains various models with a high compression rate of up to 22.6x, translating to a significantly higher compression rate for similar training performance compared to prior methods.

## Install

### 1. ActNN with AQ
+ Requirements
```bash
cd sens_act_with_aq/requirements
conda env create -f actnn_aq.yaml
```
+ Buld AAL:
```bash
conda activate actnn_aq
cd sens_act_with_aq/actnn_aq
pip install -v -e .
```

### 2. GACT with AQ
+ Requirements
```bash
cd sens_act_with_aq/requirements
conda env create -f gact_aq.yaml
```
+ Buld AAL:
```bash
conda activate gact_aq
cd sens_act_with_aq/gact_aq
pip install -v -e .
```

## Usage 

### 1. ActNN with AQ

### 2. GACT with AQ

+ Implementing ARA
```python
from aal.aal import Conv2d_ARA, Distribute_ARA

# define convolution layer wich uses ARA
self.conv1 = nn.Conv2d(64,64,3,1,1)
self.conv2 = Conv2d_ARA(64,64,3,1,1)
self.conv3 = Conv2d_ARA(64,64,3,1,1)
# define Distribute_ARA which is layer for implementing ARA
self.dsa = Distribute_ARA()

# define auxiliary residual activation for updating Conv2d_ARA
ARA = x.clone()
# doing backpropagation for conv1
x = self.conv1(x)
# adding auxiliary activation to output activation (residual connection)
# and propagating to Conv2d_ARA
x += ARA
x, ARA = self.conv2(x, ARA)
X += ARA
x, ARA = self.conv3(x, ARA)
# Distribute ARA makes self.conv2 and self.conv3 updates weight with ARA, not x!
x, ARA = self.dsa(x, ARA)
```

+ Implementing ASA
```python
from aal.aal import Linear_ASA

# define linear layer for ASA
self.linear = Linear_asa(256,256)

# propagating to Linear_ASA layer
# it would perform add 1-bit auxiliary sign activation during forward propagation
# and store this 1-bit auxiliary sign activation.
# during backprop, it would update weights by 1-bit auxiliary sign activation
x = self.linear(x)
```

## Example

[ResNet](https://github.com/asdgasdf/Average_Quantization/tree/main/benchmark/vision)

[BERT_L](https://github.com/asdgasdf/Average_Quantization/tree/main/benchmark/text_classification)

 
## Acknowledgments
  
  In this repository, code of [ActNN](https://github.com/ucbrise/actnn) and [GACT](https://github.com/LiuXiaoxuanPKU/GACT-ICML) are modified to apply with our Averge Quantization.
  Thanks the authors for open-source code.
  
 ## Lisense

> All content in this repository is licensed under the MIT license. 

