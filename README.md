# 📝 **[ACL 2025] HiddenDetect: Detecting Jailbreak Attacks against Multimodal Large Language Models via Monitoring Hidden States**

<p align="center">
  <a href="https://arxiv.org/abs/2502.14744"><strong>[📄 Arxiv]</strong></a> •
  <a href="https://huggingface.co/papers/2502.14744"><strong>[🤗 Hugging Face Daily Paper]</strong></a>
</p>

---
## 🔔 News

**[2025.05.16]**  *HiddenDetect* has been accepted to **ACL 2025 (Main)**!  🎉

---

## 🚀 Overview

Large vision-language models (LVLMs) are more vulnerable to safety risks, such as jailbreak attacks, compared to language-only models. This work explores whether LVLMs encode safety-relevant signals within their internal activations during inference. Our findings show distinct activation patterns for unsafe prompts, which can be used to detect and mitigate adversarial inputs without extensive fine-tuning.

We propose **HiddenDetect**, a tuning-free framework leveraging internal model activations to enhance safety. Experimental results demonstrate that **HiddenDetect** surpasses state-of-the-art methods in detecting jailbreak attacks against LVLMs, providing an efficient and scalable solution for robustness against multimodal threats.

---

## 📑 Contents

- [Install](#-install)  
- [Base Model](#-base-model)  
- [Dataset](#-dataset)  
- [Demo](#-demo)  
- [Citation](#-citation)

---

## ⚙️ Install

### 1. Create a virtual environment for running LLaVA
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support
pip install -e .
```

### 2. Install HiddenDetect
```bash
git clone https://github.com/leigest519/HiddenDetect.git
cd HiddenDetect
pip install -r requirements.txt
```

---

## 🏗️ Base Model

Download and save the following models to the `./model` directory:

- [llava-v1.6-vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b)  
- [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)

---
## 📂 Dataset

We evaluate the performance of **HiddenDetect** using various popular benchmark datasets, including:

| **Dataset**         | **Modality** | **Source**                                                                                                                                         | **Usage Details**                                                                                                                                                                                                                                                                                   |
|---------------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **XSTest**          | Pure text    | [XSTest](https://huggingface.co/datasets/natolambert/xstest-v2-copy)                                                                              | This dataset contains 250 safe prompts and 200 unsafe prompts.                                                                                                                                                                                                                                           |
| **FigTxt**          | Pure text    | [SafeBench CSV](https://github.com/ThuCCSLab/FigStep/blob/main/data/question/safebench.csv)                                                       | We use seven safety scenarios of this dataset, including 350 shots as unsafe samples under `./data/FigStep/safebench.csv`. These are paired with 300 handcrafted safe samples stored in `./data/FigStep/benign_questions.csv`.                                                               |
| **FigImg**          | Bimodal      | [SafeBench Images](https://github.com/ThuCCSLab/FigStep/tree/main/data/images/SafeBench)                                                           | We use all ten safety scenarios of this dataset as the visual query and pair them with the original text prompt from [FigStep](https://github.com/ThuCCSLab/FigStep) as     unsafe samples under `./data/FigStep/FigImg`.                                                                                  |
| **MM-SafetyBench**  | Bimodal      | [MM-SafetyBench](https://huggingface.co/datasets/PKU-Alignment/MM-SafetyBench)                                                                     | We include eight safety scenarios of this dataset as unsafe samples under `./data/MM-SafetyBench`.                                                                                                                                                                                                    |
| **JailbreakV-28K**  | Bimodal      | [JailbreakV-28K](https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k)                                                                    | We randomly select 300 shots belonging to five safety scenarios as unsafe samples from the `llm_transfer_attack` subset, stored under `./data/JailbreakV-28K`.                                                                                                                                  |
| **VAE**             | Bimodal      | [Visual Adversarial Examples](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models)                               | We pair four [adversarial images](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/tree/main/adversarial_images) with each prompt in the [harmful corpus](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/harmful_corpus/manual_harmful_instructions.csv) from the original repo to form a unsafe dataset stored in `./data/VAE`. |
| **MM-Vet**          | Bimodal      | [MM-Vet](https://github.com/yuweihao/MM-Vet)                                                                                                       | We use the entire dataset as safe samples under `./data/MM-Vet`. It serves as a counterpart for all bimodal unsafe datasets.                                                                                                                                                                              |

---

Since the sizes of all bimodal datasets used are different, we randomly select samples from safe and unsafe datasets to form relatively balanced evaluation datasets. This approach enhances the robustness of performance evaluation.  

Further details can be found in `./code/load_datasets.py`.

---

## 🎬 Demo

To evaluate **HiddenDetect** across all the datasets, execute:

```bash
python ./code/test.py
```

---

## 📜 Citation

If you find **HiddenDetect** or our paper helpful, please consider citing:

```bibtex
@misc{jiang2025hiddendetectdetectingjailbreakattacks,
  title={HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States},
  author={Yilei Jiang and Xinyan Gao and Tianshuo Peng and Yingshui Tan and Xiaoyong Zhu and Bo Zheng and Xiangyu Yue},
  year={2025},
  eprint={2502.14744},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2502.14744}
}
```

---

⭐ **If you like this project, give it a star!** ⭐  
💬 **Feel free to open issues** 💡
