import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# cross-entropy loss   [supervised CL paper]
class CrossModal_CL(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(CrossModal_CL, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, anchor_feature, features, label):
        # anchor_feature:[bs.1.dim]
        # features:[bs,1,dim]
        # label : [bs] long
        # b, device = anchor_feature.shape[0], anchor_feature.device
        logits = torch.div(torch.bmm(anchor_feature, features.transpose(1,2)), self.temperature)#[bs.1]
        logits_max = logits.max(dim=-1, keepdim=True)[0]
        logits = logits - logits_max.detach()

        loss = F.cross_entropy(logits.mean(dim=1), label)
        return loss

# CL loss on features
class CL_feat(nn.Module):
    def __init__(self, temperature=0.1):
        super(CL_feat, self).__init__()
        self.temperature = temperature
        self.contrastive_loss = CrossModal_CL(temperature=self.temperature)
    def forward(self, anchor, neg, pos=None):
        # anchor: [batch, x , 768]
        # pos: [batch, x, 768] 
        # neg: [batch, x, 768]
        
        if pos is not None:
            # 计算正常样本与正样本的相似度
            #print(anchor.shape)
            #print(neg.shape)
            #print(pos.shape)
            features = torch.cat([pos, neg], dim=1) #[bs,2x,512]
            labels = torch.zeros(anchor.shape[0]).long().cuda() # pos at the first
            loss = self.contrastive_loss(anchor, features, labels)
            '''sim_pos = torch.cosine_similarity(anchor, pos, dim=-1) / self.temperature
    
            # 计算负例（negative）样本对的相似性
            sim_neg = torch.cosine_similarity(anchor.unsqueeze(1), neg, dim=-1) / self.temperature
            
            # 计算InfoNCE损失
            logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)
            labels = torch.zeros(logits.size(0), dtype=torch.long).to(anchor.device)
            loss = F.cross_entropy(logits, labels)'''
        else:
            labels = torch.zeros(anchor.shape[0]).long().cuda() # pos at the first
            loss = self.contrastive_loss(anchor, neg, labels)
            '''# 计算正常样本与正样本的相似度
            pos_logits = torch.matmul(anchor, anchor.transpose(2, 1)) / self.temperature # [batch, x]
            
            # 计算正常样本与负样本的相似度
            neg_logits = torch.matmul(anchor, neg.transpose(2, 1)) / self.temperature # [batch, x]
            
            # 合并正负样本的相似度
            logits = torch.cat([pos_logits, neg_logits], dim=1) # [batch, 1+x]
            
            # 构建标签,正样本为0,负样本为1
            labels = torch.zeros(anchor.size(0), dtype=torch.long).cuda()
            
            # 计算对比损失
            loss = self.contrastive_loss(anchor, logits, labels)'''
        
        return loss
    '''def forward(self, anchor, pos, neg):
        # anchor " [bs,4,512] QV
        # pos: [bs,4,512] QV+
        # neg: [bs,x,4,512] QV-
        if neg.dim() == 3:
            neg = neg.unsqueeze(1)
        features = torch.cat([pos.unsqueeze(1), neg], dim=1) #[bs,1+x,4,512]
        label = torch.zeros(anchor.shape[0]).long().cuda() # pos at the first
        loss = []
        for i in range(4):
            loss.append(self.contrastive_loss(anchor[:,i,:], features[:,:,i,:], label))

        mean_loss = sum(loss)/len(loss)

        return mean_loss'''

if __name__ == '__main__':
    #criterion = CrossModal_CL() #SupConLoss(contrast_mode='one')
    criterion = ww_loss()
    anchor_feature = torch.rand(2,512)
    features = torch.rand(2,4,512)
    label = torch.randint(2,(2,))
    #lable = torch.ones()

    loss = criterion(anchor_feature, features, label)