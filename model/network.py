import torch
import torch.nn as nn
import torch.nn.functional as F

'''from .update import GMAUpdateBlock
from ..encoders import twins_svt_large, convnext_Xlarge_4x, convnext_base_2x
from .corr import CorrBlock, OLCorrBlock, AlternateCorrBlock
from ...utils.utils import bilinear_sampler, coords_grid, upflow8
from .gma import Attention, Aggregate'''
#import alt_cuda_corr
from torchvision.utils import save_image

autocast = torch.cuda.amp.autocast
class DirectCorr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2, coords):
        ctx.save_for_backward(fmap1, fmap2, coords)
        corr = torch.cat((fmap1, fmap2), dim=1)
        corr = torch.cat((corr, coords), dim=1)
        return corr

    def backward(ctx, grad_output):
        fmap1, fmap2, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        fmap1_grad, fmap2_grad, coords_grad = \
            alt_cuda_corr.backward(fmap1, fmap2, coords, grad_output, 4)

        return fmap1_grad, fmap2_grad, coords_grad

class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((None, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            #coords_i = (coords / 2**i).reshape(B, H, W, 2).contiguous()
            #print(fmap1_i.shape)#torch.Size([24, 224, 224, 14])

            #print(fmap2_i.shape)#torch.Size([24, 224, 224, 14])

            #print(coords_i.shape)torch.Size([112, 224, 224, 2])
            #corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            #DirectCorr.apply(fmap1_i, fmap2_i, coords_i)
            corr = torch.cat((fmap1_i, fmap2_i), dim=1)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1,1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


class MOFNet(nn.Module):
    def __init__(self):
        super().__init__()
        '''self.hidden_dim = hdim = self.cfg.feat_dim // 2
        self.context_dim = cdim = self.cfg.feat_dim // 2

        cfg.corr_radius = 4

        # feature network, context network, and update block
        if cfg.cnet == 'twins':
            print("[Using twins as context encoder]")
            self.cnet = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'basicencoder':
            print("[Using basicencoder as context encoder]")
            self.cnet = BasicEncoder(output_dim=256, norm_fn='instance')
        elif cfg.cnet == 'convnext_Xlarge_4x':
            print("[Using convnext_Xlarge_4x as context encoder]")
            self.cnet = convnext_Xlarge_4x(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'convnext_base_2x':
            print("[Using convnext_base_2x as context encoder]")
            self.cnet = convnext_base_2x(pretrained=self.cfg.pretrain)
        
        if cfg.fnet == 'twins':
            print("[Using twins as feature encoder]")
            self.fnet = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.fnet == 'basicencoder':
            print("[Using basicencoder as feature encoder]")
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance')
        elif cfg.fnet == 'convnext_Xlarge_4x':
            print("[Using convnext_Xlarge_4x as feature encoder]")
            self.fnet = convnext_Xlarge_4x(pretrained=self.cfg.pretrain)
        elif cfg.fnet == 'convnext_base_2x':
            print("[Using convnext_base_2x as feature encoder]")
            self.fnet = convnext_base_2x(pretrained=self.cfg.pretrain)

        hidden_dim_ratio = 256 // cfg.feat_dim        

        if self.cfg.Tfusion == 'stack':
            print("[Using stack.]")
            self.cfg.cost_heads_num = 1
            from .stack import SKUpdateBlock6_Deep_nopoolres_AllDecoder2
            self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder2(args=self.cfg, hidden_dim=128//hidden_dim_ratio)
        # elif self.cfg.Tfusion == 'resstack':
        #     print("[Using resstack.]")
        #     self.cfg.cost_heads_num = 1
        #     from .resstack import SKUpdateBlock6_Deep_nopoolres_AllDecoder2
        #     self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder2(args=self.cfg, hidden_dim=128)
        # elif self.cfg.Tfusion == 'stackcat':
        #     print("[Using stackcat.]")
        #     self.cfg.cost_heads_num = 1
        #     from .stackcat import SKUpdateBlock6_Deep_nopoolres_AllDecoder2
        #     self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder2(args=self.cfg, hidden_dim=128)
        

        print("[Using corr_fn {}]".format(self.cfg.corr_fn))

        gma_down_ratio = 256 // cfg.feat_dim

        self.att = Attention(args=self.cfg, dim=128//hidden_dim_ratio, heads=1, max_pos_size=160, dim_head=128//hidden_dim_ratio)

        if self.cfg.context_3D:
            print("[Using 3D Conv on context feature.]")
            self.context_3D = nn.Sequential(
                nn.Conv3d(256, 256, 3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv3d(256, 256, 3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv3d(256, 256, 3, stride=1, padding=1),
                nn.GELU(),
            )'''

    def initialize_flow(self, img, bs):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        curr_frames, H, W = img.shape
        coords0 = coords_grid(bs, H , W ).to(img.device)
        coords1 = coords_grid(bs, H , W ).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    '''def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)
    
    def upsample_flow_4x(self, flow, mask):

        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(4 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 4 * H, 4 * W)
    
    def upsample_flow_2x(self, flow, mask):

        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(2 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 2 * H, 2 * W)'''
    
    

    def forward(self, images, data={}, flow_init=None):

        #down_ratio = self.cfg.down_ratio

        B, N, channels, H, W = images.shape
        images = images.reshape(B*channels,N, H, W)
        #print(images.shape)
        '''images = 2 * (images / 255.0) - 1.0

        hdim = self.hidden_dim
        cdim = self.context_dim

        with autocast(enabled=self.cfg.mixed_precision):
            fmaps = self.fnet(images.reshape(B*N, 3, H, W)).reshape(B, N, -1, H//down_ratio, W//down_ratio)
        fmaps = fmaps.float()'''

        '''if self.cfg.corr_fn == "default":
            corr_fn = CorrBlock
        elif self.cfg.corr_fn == "efficient":
            '''
        corr_fn = AlternateCorrBlock
        forward_corr_fn = corr_fn(images[:, 1:N-1, ...], images[:, 2:N, ...], num_levels=1, radius=4)
        backward_corr_fn = corr_fn(images[:, 1:N-1, ...], images[:, 0:N-2, ...], num_levels=1, radius=4)

        '''with autocast(enabled=self.cfg.mixed_precision):
            cnet = self.cnet(images[:, 1:N-1, ...].reshape(B*(N-2), 3, H, W))
            if self.cfg.context_3D:
                #print("!@!@@#!@#!@")
                cnet = cnet.reshape(B, N-2, -1, H//2, W//2).permute(0, 2, 1, 3, 4)
                cnet = self.context_3D(cnet) + cnet
                #print(cnet.shape)
                cnet = cnet.permute(0, 2, 1, 3, 4).reshape(B*(N-2), -1, H//down_ratio, W//down_ratio)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            attention = self.att(inp)'''
        
        forward_coords1, forward_coords0 = self.initialize_flow(images[:, 0, ...], bs=B*(N-2))
        backward_coords1, backward_coords0 = self.initialize_flow(images[:, 0, ...], bs=B*(N-2))

        flow_predictions = [] # forward flows followed by backward flows

        motion_hidden_state = None

        for itr in range(12):
            
            forward_coords1 = forward_coords1.detach()
            backward_coords1 = backward_coords1.detach()

            forward_corr = forward_corr_fn(forward_coords1)
            backward_corr = backward_corr_fn(backward_coords1)
            #print(forward_corr.shape)#torch.Size([112, 6, 1, 224, 224])
            forward_flow = forward_coords1 - forward_coords0
            backward_flow = backward_coords1 - backward_coords0
            forward_flow  = forward_flow.reshape(B, N-2, 2, H, W)
            backward_flow = backward_flow.reshape(B, N-2, 2, H, W)
            #forward_flow = torch.cat((forward_flow,forward_corr),dim=2)
            #backward_flow = torch.cat((backward_flow,backward_corr),dim=2)
            '''with autocast(enabled=self.cfg.mixed_precision):
                net, motion_hidden_state, up_mask, delta_flow = self.update_block(net, motion_hidden_state, inp, forward_corr, backward_corr, forward_flow, backward_flow, forward_coords0, attention, bs=B)

            forward_up_mask, backward_up_mask = torch.split(up_mask, [down_ratio**2*9, down_ratio**2*9], dim=1)

            forward_coords1 = forward_coords1 + delta_flow[:, 0:2, ...]
            backward_coords1 = backward_coords1 + delta_flow[:, 2:4, ...]

            # upsample predictions
            if down_ratio == 4:
                forward_flow_up = self.upsample_flow_4x(forward_coords1-forward_coords0, forward_up_mask)
                backward_flow_up = self.upsample_flow_4x(backward_coords1-backward_coords0, backward_up_mask)
            elif down_ratio == 2:
                forward_flow_up = self.upsample_flow_2x(forward_coords1-forward_coords0, forward_up_mask)
                backward_flow_up = self.upsample_flow_2x(backward_coords1-backward_coords0, backward_up_mask)
            elif down_ratio == 8:
                forward_flow_up = self.upsample_flow(forward_coords1-forward_coords0, forward_up_mask)
                backward_flow_up = self.upsample_flow(backward_coords1-backward_coords0, backward_up_mask)'''

            flow_predictions.append(torch.cat([forward_flow, backward_flow], dim=1))

        #if self.training:
        return flow_predictions
        #else:
            #return flow_predictions[-1], flow_predictions[-1]
