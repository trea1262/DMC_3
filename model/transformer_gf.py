# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import pdb
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import RobertaModel, RobertaTokenizerFast
from scipy.optimize import linear_sum_assignment

class SetCriterion_SUP(nn.Module):
    """ This class computes the supervised loss for Glance-Focus.
        The process happens in two steps:
            1) we compute hungarian assignment between ground truth events and the outputs of the model
            2) we supervise each pair of matched ground-truth / prediction (supervise class and timestamp)
    """

    def __init__(self, matcher, losses, weight_dict, eos_coef, event_pred_dim):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        # classification
        empty_weight = torch.ones(event_pred_dim)
        empty_weight[-1] = eos_coef  # lower weight for background
        self.register_buffer('empty_weight', empty_weight)

    def loss_qa(self, outputs, targets, indices):
        """QA Classification Loss"""
        assert 'pred_answer' in outputs
        src_logits = outputs['pred_answer']  # (batch_size, #classes)
        target_classes = targets['qa_labels']
        loss_ce = F.cross_entropy(src_logits, target_classes)
        losses = {'loss_qa': loss_ce}
        return losses

    def loss_l1(self, outputs, targets, indices):
        """Compute the L1 regression loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center, width), normalized by the event length.
        """
        assert 'pred_spans' in outputs
        targets = targets["event_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
        losses = {}
        losses['loss_l1'] = loss_span.mean()
        return losses

    def loss_cls(self, outputs, targets, indices):
        """Event Classification loss"""
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert 'pred_logits' in outputs
        targets = targets["event_labels"]
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #events in batch
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs['pred_logits'][idx]  # (batch_size, #queries, #classes)
        target_classes = torch.cat([t['hois'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, C)
        loss_ce = F.cross_entropy(src_logits, target_classes, self.empty_weight, reduction="none")
        losses = {'loss_cls': loss_ce.mean()}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "qa": self.loss_qa,
            "l1": self.loss_l1,
            "cls": self.loss_cls
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)
        indices = self.matcher(outputs_without_aux, targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))
        return losses
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_event. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self,  cost_cls: float = 1, cost_l1: float = 1):
        """Creates the matcher

        Params:
            cost_span: This is the relative weight of the L1 error of the span coordinates in the matching cost
        """
        super().__init__()
        self.cost_cls = cost_cls
        self.cost_l1 = cost_l1
        assert cost_cls != 0 or cost_l1 != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_spans": Tensor of dim [batch_size, num_queries, 2] with the predicted span coordinates,
                    in normalized (cx, w) format
                 ""pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "spans": Tensor of dim [num_target_spans, 2] containing the target span coordinates. The spans are
                    in normalized (cx, w) format

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_spans)
        """
        bs, num_queries = outputs["pred_spans"].shape[:2]
        targets = targets['event_labels']
        # Also concat the target labels and spans
        out_prob = outputs["pred_answer"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        tgt_spans = torch.cat([v["spans"] for v in targets])  # [num_target_spans in batch, 2]
        tgt_ids = torch.cat([v["hois"] for v in targets])  # [num_target_spans in batch]
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - prob[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        #print()
        ##cost_cls = -out_prob[:, tgt_ids]  # [batch_size * num_queries, total #spans in the batch]

        # We flatten to compute the cost matrices in a batch
        out_spans = outputs["pred_spans"].flatten(0, 1)  # [batch_size * num_queries, 2]

        # Compute the L1 cost between spans
        cost_l1 = torch.cdist(out_spans, tgt_spans, p=1)  # [batch_size * num_queries, total #spans in the batch]

        # Final cost matrix
        C = self.cost_l1 * cost_l1 #+ self.cost_cls * out_prob 
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["spans"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_cls=args.set_cost_cls, cost_l1=args.set_cost_l1)


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=True,
        pass_pos_and_query=True,
        text_encoder_type="roberta-base",
        freeze_text_encoder=False,
        contrastive_loss=False,
    ):
        super().__init__()

        self.pass_pos_and_query = pass_pos_and_query
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec
        )

        self.CLS = nn.Embedding(1, d_model) if contrastive_loss else None

        self._reset_parameters()

        self.tokenizer = RobertaTokenizerFast.from_pretrained('/data/zoujy/EgoVideoQA/pretrain_model/roberta-base/')
        self.text_encoder = RobertaModel.from_pretrained('/data/zoujy/EgoVideoQA/pretrain_model/roberta-base/')

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        self.expander_dropout = 0.1
        config = self.text_encoder.config
        self.resizer = FeatureResizer(
            input_feat_size=config.hidden_size,
            output_feat_size=d_model,
            dropout=self.expander_dropout,
        )

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src=None,
        mask=None,
        query_embed=None,
        pos_embed=None,
        text=None,
        #encode_and_save=True,
        text_memory=None,
        img_memory=None,
        text_attention_mask=None,
        #glance=None
    ):
        '''if encode_and_save:
            src = src.permute(1, 0, 2)
            n, bs, c = src.shape  # (720, 12, 512)
            print(n,bs,c)
            device = src.device

            # pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            mask = mask.flatten(1)  # bs, h*w

            if self.CLS is not None:
                # We add a CLS token to the image, to be used for contrastive loss

                CLS = self.CLS.weight.view(1, 1, -1).repeat(1, bs, 1)
                # Add the CLS token to the incoming features
                src = torch.cat((CLS, src))

                # Adding zeros as the first token in the sequence to be compatible with the CLS token
                # pos_embed = torch.cat((torch.zeros(1, bs, self.d_model, device=device), pos_embed))

                # Adding one mask item to the beginning of the mask to be compatible with CLS token
                cls_pad = torch.zeros(bs, 1).bool().to(device)
                mask = torch.cat((cls_pad, mask), dim=1)

            if self.pass_pos_and_query:
                tgt = torch.zeros_like(query_embed)
            else:
                src, tgt, query_embed, pos_embed = src + 0.1 * pos_embed, query_embed, None, None

            device = src.device
            if glance:
                img_memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
                memory_cache = {
                    "img_memory": img_memory,
                    "img_pooled_op": img_memory[0] if self.CLS is not None else None,  # Return the CLS token
                    "mask": mask,
                    "pos_embed": pos_embed,
                    "query_embed": query_embed,
                }
                return memory_cache
            else:
                if isinstance(text[0], str):
                    # Encode the text
                    tokenized = self.tokenizer.batch_encode_plus(text, padding="longest", max_length=100, return_tensors="pt").to(device)
                    encoded_text = self.text_encoder(**tokenized)

                    # Transpose memory because pytorch's attention expects sequence first
                    text_memory = encoded_text.last_hidden_state.transpose(0, 1)
                    # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
                    text_attention_mask = tokenized.attention_mask.ne(1).bool()

                    # Resize the encoder hidden states to be of the same d_model as the decoder
                    text_memory_resized = self.resizer(text_memory)
                else:
                    # The text is already encoded, use as is.
                    text_attention_mask, text_memory_resized, tokenized = text
                # pdb.set_trace()

                # Concat on the sequence dimension
                src = torch.cat([src, text_memory_resized], dim=0)
                # For mask, sequence dimension is second
                mask = torch.cat([mask, text_attention_mask], dim=1)
                # Pad the pos_embed with 0 so that the addition will be a no-op for the text tokens
                # pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory_resized)], dim=0)
                # pdb.set_trace()
                img_memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

                text_memory = img_memory[-len(text_memory_resized):]
                # num_events = 10
                # event_memory = img_memory[-len(text_memory_resized)-num_events:-len(text_memory_resized)]

                assert img_memory.shape[1] == text_memory.shape[1] == tgt.shape[1]
                memory_cache = {
                    "text_memory_resized": text_memory_resized,
                    "text_memory": text_memory,
                    "img_memory": img_memory,
                    "text_pooled_op": encoded_text.pooler_output if self.CLS is not None else None,
                    "img_pooled_op": img_memory[0] if self.CLS is not None else None,  # Return the CLS token
                    "mask": mask,
                    "text_attention_mask": text_attention_mask,
                    "pos_embed": pos_embed,
                    "query_embed": query_embed,
                    "tokenized": tokenized,
                }
                return memory_cache

        else:
            if self.pass_pos_and_query:
                tgt = torch.zeros_like(query_embed)
            else:
                src, tgt, query_embed, pos_embed = src + 0.1 * pos_embed, query_embed, None, None

            # assert img_memory.shape[1] == text_memory.shape[1] == tgt.shape[1]
            # pdb.set_trace()'''
        hs = self.decoder(
                tgt,
                img_memory,
                text_memory,
                memory_key_padding_mask=mask,
                text_memory_key_padding_mask=text_attention_mask,
                pos=pos_embed,
                query_pos=query_embed,
                glance=glance
            )
        print(hs.transpose(1, 2).shape)
        return hs.transpose(1, 2)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):

        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        glance=None
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                text_memory=text_memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                text_memory_key_padding_mask=text_memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                glance=glance
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.cross_attn_text = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # For now, trying one version where its self attn -> cross attn text -> cross attn image -> FFN
    def forward_post(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)

        # Self attention
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention to text
        # tgt2 = self.cross_attn_text(
        #     query=self.with_pos_embed(tgt, query_pos),
        #     key=text_memory,
        #     value=text_memory,
        #     attn_mask=None,
        #     key_padding_mask=text_memory_key_padding_mask,
        # )[0]
        # tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)

        # Cross attention to image
        tgt2 = self.cross_attn_image(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt

    # Multi-Level Attention
    def forward_MLA(
        self,
        tgt,  # (num_queries, B, D)
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)

        # Self attention
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # pdb.set_trace()
        img_memory, event_memory = memory[:-10], memory[-10:]

        # Cross attention to question
        selected_event_memory = self.cross_attn_image(
            query=text_memory,
            key=event_memory,
            value=event_memory,
            attn_mask=None,
            key_padding_mask=memory_key_padding_mask[:, -10:],
        )[0]
        selected_event_memory = self.dropout2(selected_event_memory)  # (10, B, D)
        selected_event_memory = self.norm2(selected_event_memory)

        # Cross attention to image
        selected_image_memory = self.cross_attn_image(
            query=selected_event_memory,
            key=img_memory,
            value=img_memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask[:, :-10],
        )[0]
        selected_image_memory = self.dropout3(selected_image_memory)  # (N, B, D)
        selected_image_memory = self.norm3(selected_image_memory)

        # Cross attention to image
        # pdb.set_trace()
        tgt2 = self.cross_attn_image(
            query=self.with_pos_embed(tgt, query_pos),
            key=selected_image_memory,
            value=selected_image_memory,
            attn_mask=memory_mask,
            key_padding_mask=text_memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout5(tgt2)
        tgt = self.norm5(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        glance=False
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos
            )
        if not glance:
            return self.forward_MLA(
                tgt,
                memory,
                text_memory,
                tgt_mask,
                memory_mask,
                text_memory_key_padding_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            text_memory,
            tgt_mask,
            memory_mask,
            text_memory_key_padding_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        num_encoder_layers=768,
        num_decoder_layers=768
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
