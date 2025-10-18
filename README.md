# FDA
This repository is the official PyTorch implementation of “Model Merging with Functional Dual Anchors”, by Kexuan Shi, Yandong Wen, Weiyang Liu.

([arxiv](https://arxiv.org/pdf/2401.07402.pdf))

![IGA-INR demo](figures/visualization.gif)


## Introduction
Model Merging has been an intriguing post-training strategy for integrating knowledge from mutliple finetuned checkpoints of a shared foundation model. Existing methods focuses on the operation in the parameter space, i.e, combing task vectors to mitgate knowledge confilctsm, thereby remain constrained by the complexity of the parameter space. In this work, we propose Functional Dual Anchors (FDAs), a framework that instead models the knowledge in the input-representation space. Specifically, FDAs are synthetic inputs whose induced gradients align with task vectors, capturing task-specific functional shifts relative to the pretrained model. Then, we use the FDAs to adapt the pretrained model. Comparing with the task vectors, FDAs can provide more robust and flexible trajectory for model merging, as shown in the Figure.
FDAs provide an alternative perspective on model merging by extending input-space modeling to this setting and bridges joint multi-task training and post-hoc merging.

## Quick Start

## Construct FDAs

## Acknowledgement

## Citation
