# DMC_3
[ACM MM 25]Code for the paper:  DMC³: Dual-Modal Counterfactual Contrastive Construction for Egocentric Video Question Answering  Accepted to the ACM International Conference on Multimedia (ACM MM) 2025.

## 📝 Preparation
### 1. Install Dependencies 
Installs dependencies needed for the code to run.
```bash
conda create -n egovqa python=3.9 pip
pip install torch-1.12.1+cu113-cp39-cp39-linux_x86_64.whl
pip install torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

### 2. Data Download
You can get the dataset by following the data processing steps provided in the [EgoTaskQA](https://github.com/Buzz-Beater/EgoTaskQA/blob/main/baselines/README.md) work. Also, you can download the processed data directly by following the [EgoVideoQA](https://github.com/Hyu-Zhang/EgoVideoQA/blob/main/README.md) work.
```
wget https://drive.google.com/file/d/1TMJ3qcMt-psDuevw4JaXd7pOzwmMk6wR/view?usp=sharing
tar -zxvf Data.tar.gz && rm Data.tar.gz
# The following links are provided by EgoVLPv2, see https://github.com/facebookresearch/EgoVLPv2/tree/main/EgoTaskQA
wget https://www.cis.jhu.edu/~shraman/EgoVLPv2/datasets/EgoTaskQA/qa_videos.tgz
tar -xvzf qa_videos.tgz && rm qa_videos.tgz
```

### 3. Pretrained Weights
We use the EgoVLPv2 model weights, which are pre-trained on the [EgoClip](https://drive.google.com/file/d/1-aaDu_Gi-Y2sQI_2rsI2D1zvQBJnHpXl/view?usp=sharing) version of [Ego4D](https://ego4d-data.org/docs/start-here/#cli-download). And you can follow the commands below.
```
wget -c https://www.cis.jhu.edu/~shraman/EgoVLPv2/ckpts/Pre-trained/EgoVLPv2.pth
wget https://www.cis.jhu.edu/~shraman/EgoVLPv2/datasets/EgoTaskQA/reasoning_unique_cat.pth
# ViT from timm package
wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
# RoBERTa from huggingface
https://huggingface.co/roberta-base/tree/main
```

## 🔧 Fine-tuning
Modify the target path parameters, including ```writer```, ```--basedir```, ```--model_name```, ```data_dir```, ```meta_dir```, and ```unique_dict``` in the file ```main_end2end.py```, ```metadata_dir```, ```unique_dict```, and ```tokenizer``` in the file ```EgoTaskQA_dataset.py```, and ```self.text_model``` and ```vit_model``` in the file ```/model/video_qa_model_linear_end2end.py```.

After that, you can fine-tune model on EgoTaskQA dataset with the following commands. The split type is controlled by the ```--dataset_split_type``` argument.
The training process of our model is divided into two stages. 
```
# first stage
# direct setting
python main.py --dataset_split_type direct --model_name /userhome/pretrain_model/EgoVLPv2.pth --per_gpu_batch_size 32 --num_frames_per_video 16 --frame_resolution 224 --lr 2e-4
# indirect setting
python main.py --dataset_split_type indirect --model_name /userhome/pretrain_model/EgoVLPv2.pth --per_gpu_batch_size 32 --num_frames_per_video 16 --frame_resolution 224 --lr 2e-4
```
The model parameters obtained after the first stage of fine-tuning are used as input for the pre-training parameters in the second stage.
```
# second stage
python mainneg.py --dataset_split_type direct --model_name <first_stage_saved_ckpth> --per_gpu_batch_size 32 --num_frames_per_video 16 --frame_resolution 224 --lr 2e-4
```

## 🎯 Evaluation
To evaluate the fine-tuned checkpoints, you are able to add ```--test_only_model_path``` argument.
```
python mainneg.py --dataset_split_type direct --test_only_model_path <model_best_ckpt> --per_gpu_batch_size 32 --num_frames_per_video 16 --frame_resolution 224 --lr 2e-4
```

## 🏆 Fine-tuned Checkpoints
We will provide our fine-tuned model checkpoints as soon as possible.
https://drive.google.com/drive/folders/1fUoKMetg2ehEFbuwZDem4HQRuw6bZxcb?usp=sharing

## 🎓 Citation
If our work is helpful to you, please cite our paper.

```
@inproceedings{10.1145/3746027.3755085,
author = {Zou, Jiayi and Chen, Chaofan and Bao, Bing-Kun and Xu, Changsheng},
title = {DMC3: Dual-Modal Counterfactual Contrastive Construction for Egocentric Video Question Answering},
year = {2025},
booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
pages = {3438–3447}
```

## ✉️ Contact
Questions and discussions are welcome via `2023010213@njupt.edu.cn`.

## 🙏 Acknowledgements
We thank the authors from [EgoTaskQA](https://github.com/Buzz-Beater/EgoTaskQA/tree/main) for releasing the dataset and baselines. Also, we thank the authors from [EgoVLP](https://github.com/showlab/EgoVLP?tab=readme-ov-file), [EgoVLPv2](https://github.com/facebookresearch/EgoVLPv2/tree/main) and [EgoVideoQA](https://github.com/Hyu-Zhang/EgoVideoQA/blob/main/README.md) for the exploratory research, which is the beginning of our study.

## 🔖 License
[MIT License]()
