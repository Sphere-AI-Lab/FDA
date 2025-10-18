# FDA
This repository is the official PyTorch implementation of “Model Merging with Functional Dual Anchors”, by Kexuan Shi, Yandong Wen, Weiyang Liu.

([arxiv](https://arxiv.org/pdf/2401.07402.pdf))

<p align="center">
  <img src="docs/assets/framework_trajectory.png" width="90%" />
</p>


## Introduction
Model Merging has been an intriguing post-training strategy for integrating knowledge from mutliple finetuned checkpoints of a shared foundation model. Existing methods focuses on the operation in the parameter space, i.e, combing task vectors to mitgate knowledge confilctsm, thereby remain constrained by the complexity of the parameter space. In this work, we propose **Functional Dual Anchors (FDAs)**, a framework (Figure 1(a)) that instead models the knowledge in the input-representation space. Specifically, FDAs are synthetic inputs whose induced gradients align with task vectors, capturing task-specific functional shifts relative to the pretrained model. Then, we use the FDAs to adapt the pretrained model. Comparing with the task vectors, FDAs can provide more robust and flexible trajectory for model merging, as shown in the Figure 1(b).
FDAs provide an alternative perspective on model merging by extending input-space modeling to this setting and bridges joint multi-task training and post-hoc merging.

## Quick Start
### Checkpoints and corresponding FDAs
To help you quickly get started with FDAs, we provide download links for the checkpoints used in the paper, along with the corresponding FDAs. 

For Vision tasks, we simply use the checkpoints from the link: https://drive.google.com/drive/folders/1u_Tva6x0p6oxu5Eo0ZZsf-520Cc_3MKw. 

For NLP tasks, we adopt the pretrained RoBERTa-base and RoBERTa-lagre from https://huggingface.co/FacebookAI/roberta-large. Then, we use the finetuning script from DARE to obtain the checkpoints on eight GLUE Benchmarks. The links are ....

For NLG tasks, the pretrained model is downloaded from https://huggingface.co/meta-llama/Llama-2-13b-hf. The expert model in Math and Code are from https://huggingface.co/vanillaOVO/WizardMath-13B-V1.0 and https://huggingface.co/layoric/llama-2-13b-code-alpaca.

The FDAs for the above models can be found in: .

### Environment
For Vision and NLP tasks, we use the same environment. It can be installed by:
```bash
cd FDA/Vision #cd FDA/NLU
# Create conda environment
conda env create -f environment.yaml
# Activate environment
conda activate fda
```
For NLU tasks, please use: ```NLG/environment.yaml```

### Adapt by FDAs
Please follow the path comments in the code file ```adapt.py```, replace them with the paths to your local checkpoints and FDAs, and then run the following commands to reproduce the FDA adaptation results:
```bash
cd FDA/Vision #cd FDA/NLU cd FDA/NLG
sh adapt.sh
```

For models in NLG tasks, please split the model first:
```bash
cd FDA/NLG
python split_model.py
```

### Construct FDAs
If you want to construct FDAs for your finetuned checkpoint, please follow the path comments in the code file ```construct_fda.py```, replace them with the paths to your finetuned checkpoints. Then,
```bash
sh construct.sh
```

## Acknowledgement
This repository uses codes and resources from Task Arithmetic, DARE, TSVM, WUDI, Prodistill.

## Citation
