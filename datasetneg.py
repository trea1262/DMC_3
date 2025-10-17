# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import torch
import pandas as pd
import pdb
from base.base_dataset import TextVideoDataset
# from data_loader.transforms import init_transform_dict, init_video_transform_dict
from transforms import init_transform_dict, init_video_transform_dict
import json
import transformers
from PIL import Image
import torch
from transforms import transforms

class EgoTaskQA(TextVideoDataset):
    def _load_metadata(self, args):
        metadata_dir = '.../data/Data/qa_ori/' + args.dataset_split_type
        split_files = {
            'train': 'formatted_train_qas_encode4.json',
            'val': 'formatted_val_qas_encode4.json',       
            'test': 'formatted_test_qas_encode4.json',
        }
        target_split_fp = split_files[self.split]

        with open(os.path.join(metadata_dir, target_split_fp),'r') as load_f:
            metadata = json.load(load_f)
        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        self.metadata = pd.DataFrame(metadata)

        if self.split in ['train']:
            self.frame_sample = 'rand'
        elif self.split in ['val', 'test']:
            self.frame_sample = 'uniform'

    def _get_video_path(self, sample):
        #sample['interval'] = sample['interval'].replace('|', '_')
        rel_video_fp = sample['interval'] + '.mp4'
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return full_video_fp

    def _get_video_frames(self, video_fp):
        video_loading = self.video_params.get('loading', 'strict')
        # pdb.set_trace()
        try:
            if os.path.isfile(video_fp):
                imgs, idxs = self.video_reader(video_fp, self.video_params['num_frames'])
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == 'strict':
                raise ValueError(
                    f'Video loading failed for {video_fp}, video loading for this dataset is strict.') from e
            else:
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = transforms.ToTensor()(imgs).unsqueeze(0)

        if self.transforms is not None:
            if self.video_params['num_frames'] > 1:
                imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
                imgs = self.transforms(imgs)
                imgs = imgs.transpose(0, 1)  # recover
            else:
                imgs = self.transforms(imgs)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'], self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs
        return final
    
    def apply_center_mask(self,frames):
        """
        对输入的视频帧序列实现中间部分的 mask 操作
        
        Args:
            frames (torch.Tensor): 输入的视频帧序列, 形状为 [num_frames, channels, height, width]
        
        Returns:
            torch.Tensor: 经过中间部分 mask 处理后的视频帧序列
        """
        num_frames, channels, height, width = frames.shape
        
        # 计算 mask 的尺寸
        mask_height = height // 2
        mask_width = width // 2
        
        # 创建中间部分的 mask
        mask = torch.zeros_like(frames)
        mask[:, :, (height-mask_height)//2:(height+mask_height)//2, (width-mask_width)//2:(width+mask_width)//2] = 1#(height+mask_height)//2
        
        # 应用 mask 到输入的视频帧序列
        masked_frames = frames * mask
        
        return masked_frames
    def apply_outer_mask(self,frames):
        """
        对输入的视频帧序列实现周围部分的 mask 操作
        
        Args:
            frames (torch.Tensor): 输入的视频帧序列, 形状为 [num_frames, channels, height, width]
        
        Returns:
            torch.Tensor: 经过周围部分 mask 处理后的视频帧序列
        """
        num_frames, channels, height, width = frames.shape
        
        # 计算 mask 的尺寸
        mask_height = height // 4
        mask_width = width // 4
        
        # 创建周围部分的 mask
        mask = torch.zeros_like(frames)
        mask[:, :, :mask_height, :] = 1  # 上部分
        mask[:, :, -mask_height:, :] = 1  # 下部分
        mask[:, :, mask_height:-mask_height, :mask_width] = 1  # 左部分
        mask[:, :, mask_height:-mask_height, -mask_width:] = 1  # 右部分
        
        # 应用 mask 到输入的视频帧序列
        masked_frames = frames * mask
        #masked_frames = masked_frames.flip([0])
        
        return masked_frames

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp  = self._get_video_path(sample)

        final = self._get_video_frames(video_fp)
        video_positive = self.apply_center_mask(final)
        video_negative = self.apply_outer_mask(final)

        #pdb.set_trace()
        meta_arr = {'type': sample['type'], 'category': sample['category'], 'semantic': sample['semantic'], 'reasoning': sample['reasoning_type'].split('$')}
        return {'video': final, 'video_pos':video_positive, 'video_neg': video_negative, 'text': sample['question'],'text_neg': sample['text_neg'],'text_pos': sample['text_pos'], 'meta': meta_arr, 'answer': sample['answer_encode'], 'sample':sample['sample']}

def collate_func(batch):
    
    config = '/data/zoujy/EgoVideoQA/pretrain_model/roberta-base/config.json'
    unique_dict = torch.load('/data/zoujy/EgoVideoQA/pretrain_model/reasoning_unique_cat.pth')
    f = open(config)
    config = json.load(f)
    tokenizer = transformers.AutoTokenizer.from_pretrained("/data/zoujy/EgoVideoQA/pretrain_model/roberta-base/", local_files_only=True)
    video_frames_list, mask_video_frames_list, video_center_frames_list, text_tokens_list, attention_masks_list,textneg_tokens_list, attentionneg_masks_list,textpos_tokens_list, attentionpos_masks_list, answers_list, reasoning_list, sample_tokens_list, sample_attention_masks_list = [], [], [], [], [], [], [], [], [], [], [], [], []
                            
    for i, item in enumerate(batch):
        video_frames_list.append(item['video'])
        mask_video_frames_list.append(item['video_pos'])
        video_center_frames_list.append(item['video_neg'])
        tokenized_text = tokenizer(item['text'].strip().lower(), return_tensors='pt', padding='max_length', max_length=30, truncation=True)
        text_tokens_list.append(tokenized_text['input_ids'])
        attention_masks_list.append(tokenized_text['attention_mask'])

        tokenizedneg_text = tokenizer(item['text_neg'].strip().lower(), return_tensors='pt', padding='max_length', max_length=30, truncation=True)
        textneg_tokens_list.append(tokenizedneg_text['input_ids'])
        attentionneg_masks_list.append(tokenizedneg_text['attention_mask'])
        tokenizedpos_text = tokenizer(item['text_pos'].strip().lower(), return_tensors='pt', padding='max_length', max_length=30, truncation=True)
        textpos_tokens_list.append(tokenizedpos_text['input_ids'])
        attentionpos_masks_list.append(tokenizedpos_text['attention_mask'])
        answers_list.append(torch.tensor(item['answer']))

        index_list = []
        for _idx in range(len(item['meta']['reasoning'])):
            index_list.append(unique_dict[item['meta']['reasoning'][_idx]])
        reasoning_list.append(torch.tensor(index_list))

        #sample_list.append(item['sample'])
        combined_string = " ".join(item['sample'])
        tokenized_sample = tokenizer(combined_string.strip().lower(), return_tensors='pt', padding='max_length', max_length=30, truncation=True)
        sample_tokens_list.append(tokenized_sample['input_ids'])
        sample_attention_masks_list.append(tokenized_sample['attention_mask'])

    video_frames = torch.stack(video_frames_list, dim=0)
    mask_video_frames = torch.stack(mask_video_frames_list, dim=0)
    video_center_frames = torch.stack(video_center_frames_list, dim=0)
    text_tokens = torch.stack(text_tokens_list, dim=0).squeeze(1)
    attention_masks = torch.stack(attention_masks_list, dim=0).squeeze(1)
    textneg_tokens = torch.stack(textneg_tokens_list, dim=0).squeeze(1)
    attentionneg_masks = torch.stack(attentionneg_masks_list, dim=0).squeeze(1)
    textpos_tokens = torch.stack(textpos_tokens_list, dim=0).squeeze(1)
    attentionpos_masks = torch.stack(attentionpos_masks_list, dim=0).squeeze(1)
    answers = torch.stack(answers_list, dim=0)

    samples = torch.stack(sample_tokens_list,dim=0)
    sample_attention_masks_list = torch.stack(sample_attention_masks_list, dim=0)
    # # torch.Size([4, 20, 3, 224, 224]) torch.Size([4, 12]) torch.Size([4]) torch.Size([4, 20, 2048]) torch.Size([4, 25, 15])
    return video_frames, mask_video_frames, video_center_frames, text_tokens, attention_masks, textneg_tokens, attentionneg_masks, textpos_tokens, attentionpos_masks, answers, reasoning_list#, samples, sample_attention_masks_list

if __name__ == "__main__":
    kwargs = dict(
        dataset_name="EgoTaskQA",
        text_params={
            "input": "text"
        },
        video_params={
        "input_res": 224,
        "num_frames": 4,
        "loading": "lax"
        },
        data_dir="EgoTaskQA/qa_videos",
        meta_dir="Data/qa/indirect",
        tsfms=init_video_transform_dict()['test'],
        reader='decord',
        split='train',
        neg_param=60
    )
    dataset = EgoTaskQA(**kwargs)
    import tqdm
    max_class = 0
    for i in tqdm.tqdm(range(len(dataset))):
        item = dataset[i]
        pdb.set_trace()
        class_no = item['answer']
        if class_no > max_class:
            max_class = class_no
        # print(item.keys())
    print(max_class)
