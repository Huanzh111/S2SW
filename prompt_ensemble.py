import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np
from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
# from open_clip import tokenizer
# simple_tokenizer = tokenizer.SimpleTokenizer()
from copy import deepcopy
import torch.nn as nn

_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[
    torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def encode_text_with_prompt_ensemble(model, texts, device):
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect',
                     '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.',
                        'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.',
                        'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.',
                        'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
                        'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.',
                        'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.',
                        'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.',
                        'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.',
                        'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.',
                        'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.',
                        'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.',
                        'this is the {} in the scene.', 'this is one {} in the scene.']

    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(texts[0]) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence)
        class_embeddings = model.encode_text(prompted_sentence.to(device))
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        text_features.append(class_embedding)

    text_features = torch.stack(text_features, dim=1).to(device).t()

    return text_features


def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])


class AnomalyCLIP_PromptLearner(nn.Module):
    def __init__(self, clip_model, design_details):
        super().__init__()
        classnames = ["object"]
        self.n_cls = len(classnames)  # self.n_cls:  1
        self.n_ctx = design_details["Prompt_length"]  # self.n_ctx:  12
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.n_ctx
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"]  # self.text_encoder_n_ctx:  4
        ctx_init_pos = ""
        ctx_init_neg = ""
        dtype = clip_model.transformer.get_cast_dtype()  # dtype:  torch.float32
        ctx_dim = clip_model.ln_final.weight.shape[0]  # ctx_dim:  768
        self.classnames = classnames
        self.state_normal_list = [
            "{}",
        ]
        self.state_anomaly_list = [
            "damaged {}",
        ]
        normal_num = len(self.state_normal_list)
        anormaly_num = len(self.state_anomaly_list)
        self.normal_num = normal_num
        self.anormaly_num = anormaly_num  # self.normal_num:  1 self.anormaly_num:  1

        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            # 初始化text成bpd编码
            prompt_pos = tokenize(ctx_init_pos)
            prompt_neg = tokenize(ctx_init_neg)
            with torch.no_grad():
                # 生成相应的text embedding
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            # 这些是去除出来EOS 和 # CLS, EOS， 获得可学习的textual prompt
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if True:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(self.n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)
        else:
            # Random Initialization
            if True:
                print("Initializing class-specific contexts")
                # 这里是cls是类的个数，n_ctx_pos代表learnable token的长度，ctx_dim表示prompt的dimension
                ctx_vectors_pos = torch.empty(self.n_cls, self.normal_num, n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(self.n_cls, self.anormaly_num, n_ctx_neg, ctx_dim, dtype=dtype)
                # ctx_vectors_pos:  torch.Size([1, 1, 12, 768])
                # ctx_vectors_neg:  torch.Size([1, 1, 12, 768])
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                       for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            # single_para torch.Size([4, 768])
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(ctx_dim, 896)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized
        classnames = [name.replace("_", " ") for name in classnames]

        key_words_pos_class = "perfectly good normal intact clean healthy flawless perfect ideal standard"
        key_words_neg_class = "crack scratch dent stain discoloration bubble corrosion flaking deformation hole loose scar foam"

        key_words_pos_emo = "perfectly pristine flawless perfect intact consistent uniform normal stable standard well ideal"
        key_words_neg_emo = "severely distorted flawed warped broken cracked uneven anomalous irregular aberrant unusual deformed"
        prompt_pos_emo = [prompt_prefix_pos + " " + key_words_pos_emo + "."]
        prompt_neg_emo = [prompt_prefix_neg + " " + key_words_neg_emo + "."]
        prompt_pos_class = [prompt_prefix_pos + " " + key_words_pos_class + "."]
        prompt_neg_class = [prompt_prefix_neg + " " + key_words_neg_class + "."]
        prompts_pos = [prompt_prefix_pos + " " + template.format(name) + "." for template in self.state_normal_list for
                       name in classnames]
        # prompts_pos:  ['X X X X X X X X X X X X object.']
        prompts_neg = [prompt_prefix_neg + " " + template.format(name) + "." for template in self.state_anomaly_list for
                       name in classnames]
        # prompts_neg:  ['X X X X X X X X X X X X damaged object.']
        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
        tokenized_prompts_pos_emo = []
        tokenized_prompts_neg_emo = []
        tokenized_prompts_pos_class = []
        tokenized_prompts_neg_class = []
        for p_pos_class in prompt_pos_class:
            tokenized_prompts_pos_class.append(tokenize(p_pos_class))
        for p_neg_class in prompt_neg_class:
            tokenized_prompts_neg_class.append(tokenize(p_neg_class))
        for p_pos_emo in prompt_pos_emo:
            tokenized_prompts_pos_emo.append(tokenize(p_pos_emo))
        for p_neg_emo in prompt_neg_emo:
            tokenized_prompts_neg_emo.append(tokenize(p_neg_emo))
        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(tokenize(p_pos))
        for p_neg in prompts_neg:
            tokenized_prompts_neg.append(tokenize(p_neg))
        tokenized_prompts_pos_class = torch.cat(tokenized_prompts_pos_class)
        tokenized_prompts_neg_class = torch.cat(tokenized_prompts_neg_class)
        tokenized_prompts_pos_emo = torch.cat(tokenized_prompts_pos_emo)
        tokenized_prompts_neg_emo = torch.cat(tokenized_prompts_neg_emo)
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        # [49406,   343,   343,   343,   343, 343,   343,   343, 343,343, 343,   343,   343, 14115,   269, 49407
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        # [49406,  343,  343, 343, 343, 343, 343,  343,   343,   343, 343,  343,   343, 13568, 14115,   269, 49407
        # 生成相应的text embedding
        with torch.no_grad():
            embedding_pos_class = clip_model.token_embedding(tokenized_prompts_pos_class).type(dtype)
            embedding_neg_class = clip_model.token_embedding(tokenized_prompts_neg_class).type(dtype)
            embedding_pos_emo = clip_model.token_embedding(tokenized_prompts_pos_emo).type(dtype)
            embedding_neg_emo = clip_model.token_embedding(tokenized_prompts_neg_emo).type(dtype)
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)  # [1, 77, 768]
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)  # [1, 77, 768]
            n, l, d = embedding_pos.shape
            n1, l1, d1 = embedding_pos_emo.shape
            n2, l2, d2 = embedding_pos_class.shape
            embedding_pos_class = embedding_pos_class.reshape(normal_num, self.n_cls, l2, d2).permute(1, 0, 2, 3)
            embedding_neg_class = embedding_neg_class.reshape(anormaly_num, self.n_cls, l2, d2).permute(1, 0, 2, 3)
            embedding_pos_emo = embedding_pos_emo.reshape(normal_num, self.n_cls, l1, d1).permute(1, 0, 2, 3)
            embedding_neg_emo = embedding_neg_emo.reshape(anormaly_num, self.n_cls, l1, d1).permute(1, 0, 2, 3)
            embedding_pos = embedding_pos.reshape(normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)  # [1, 1, 77, 768]
            embedding_neg = embedding_neg.reshape(anormaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)  # [1, 1, 77, 768]

        # 我们可能希望模型中的某些参数参数不更新（从开始到结束均保持不变）
        self.register_buffer("token_prefix_pos_class", embedding_pos_class[:, :, :1, :])  # torch.Size([1, 1, 1, 768])
        self.register_buffer("token_suffix_pos_class", embedding_pos_class[:, :, 1 + n_ctx_pos:, :])  # torch.Size([1, 1, 64, 768])
        self.register_buffer("token_prefix_neg_class", embedding_neg_class[:, :, :1, :])  # torch.Size([1, 1, 1, 768])
        self.register_buffer("token_suffix_neg_class", embedding_neg_class[:, :, 1 + n_ctx_neg:, :])  # torch.Size([1, 1, 64, 768])

        self.register_buffer("token_prefix_pos_emo", embedding_pos_emo[:, :, :1, :])  # torch.Size([1, 1, 1, 768])
        self.register_buffer("token_suffix_pos_emo", embedding_pos_emo[:, :, 1 + n_ctx_pos:, :])  # torch.Size([1, 1, 64, 768])
        self.register_buffer("token_prefix_neg_emo", embedding_neg_emo[:, :, :1, :])  # torch.Size([1, 1, 1, 768])
        self.register_buffer("token_suffix_neg_emo", embedding_neg_emo[:, :, 1 + n_ctx_neg:, :])  # torch.Size([1, 1, 64, 768])

        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :])  # torch.Size([1, 1, 1, 768])
        self.register_buffer("token_suffix_pos", embedding_pos[:, :, 1 + n_ctx_pos:, :])  # torch.Size([1, 1, 64, 768])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :, :1, :])  # torch.Size([1, 1, 1, 768])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_neg:, :])  # torch.Size([1, 1, 64, 768])

        n2, d2 = tokenized_prompts_pos_class.shape
        tokenized_prompts_pos_class = tokenized_prompts_pos_class.reshape(normal_num, self.n_cls, d2).permute(1, 0, 2)
        n1, d1 = tokenized_prompts_pos_emo.shape
        tokenized_prompts_pos_emo = tokenized_prompts_pos_emo.reshape(normal_num, self.n_cls, d1).permute(1, 0, 2)
        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(normal_num, self.n_cls, d).permute(1, 0, 2)
        # torch.Size([1, 1, 77])

        n2, d2 = tokenized_prompts_neg_class.shape
        tokenized_prompts_neg_class = tokenized_prompts_neg_class.reshape(anormaly_num, self.n_cls, d2).permute(1, 0, 2)
        n1, d1 = tokenized_prompts_neg_emo.shape
        tokenized_prompts_neg_emo = tokenized_prompts_neg_emo.reshape(anormaly_num, self.n_cls, d1).permute(1, 0, 2)
        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(anormaly_num, self.n_cls, d).permute(1, 0, 2)
        # torch.Size([1, 1, 77])
        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        # 我们可能希望模型中的某些参数参数不更新（从开始到结束均保持不变）
        self.register_buffer("tokenized_prompts_pos_class", tokenized_prompts_pos_class)
        self.register_buffer("tokenized_prompts_neg_class", tokenized_prompts_neg_class)

        self.register_buffer("tokenized_prompts_pos_emo", tokenized_prompts_pos_emo)
        self.register_buffer("tokenized_prompts_neg_emo", tokenized_prompts_neg_emo)

        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)

    def forward(self, cls_id=None):

        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        # print("shape", self.ctx_pos[0:1].shape, ctx_pos.shape)
        prefix_pos_class = self.token_prefix_pos_class
        prefix_neg_class = self.token_prefix_neg_class
        suffix_pos_class = self.token_suffix_pos_class
        suffix_neg_class = self.token_suffix_neg_class

        prefix_pos_emo = self.token_prefix_pos_emo
        prefix_neg_emo = self.token_prefix_neg_emo
        suffix_pos_emo = self.token_suffix_pos_emo
        suffix_neg_emo = self.token_suffix_neg_emo

        prefix_pos = self.token_prefix_pos
        prefix_neg = self.token_prefix_neg
        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg
        # torch.Size([1, 1, 1, 768]) torch.Size([1, 1, 12, 768]) torch.Size([1, 1, 64, 768])
        prompts_pos_class = torch.cat(
            [
                # N(the number of template), 1, dim
                prefix_pos_class,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos_class,  # (n_cls, *, dim)
            ], dim=2, )
        prompts_neg_class = torch.cat(
            [
                prefix_neg_class,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg_class,  # (n_cls, *, dim)
            ], dim=2, )

        prompts_pos_emo = torch.cat(
            [
                # N(the number of template), 1, dim
                prefix_pos_emo,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos_emo,  # (n_cls, *, dim)
            ], dim=2,)
        prompts_neg_emo = torch.cat(
            [
                prefix_neg_emo,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg_emo,  # (n_cls, *, dim)
            ], dim=2,)
        prompts_pos = torch.cat(
            [
                # N(the number of template), 1, dim
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos,  # (n_cls, *, dim)
            ], dim=2,)
        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ], dim=2,)
        _, _, l2, d2 = prompts_pos_class.shape
        prompts_pos_class = prompts_pos_class.reshape(-1, l2, d2)
        _, _, l2, d2 = prompts_neg_class.shape
        prompts_neg_class = prompts_neg_class.reshape(-1, l2, d2)
        prompts_class = torch.cat([prompts_pos_class, prompts_neg_class], dim=0)

        _, _, l1, d1 = prompts_pos_emo.shape
        prompts_pos_emo = prompts_pos_emo.reshape(-1, l1, d1)
        _, _, l1, d1 = prompts_neg_emo.shape
        prompts_neg_emo = prompts_neg_emo.reshape(-1, l1, d1)
        prompts_emo = torch.cat([prompts_pos_emo, prompts_neg_emo], dim=0)

        _, _, l, d = prompts_pos.shape
        prompts_pos = prompts_pos.reshape(-1, l, d)
        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d)
        prompts = torch.cat([prompts_pos, prompts_neg], dim=0)

        # torch.Size([1, 77, 768]) torch.Size([1, 77, 768]) torch.Size([2, 77, 768])
        _, l2, d2 = self.tokenized_prompts_pos_class.shape
        tokenized_prompts_pos_class = self.tokenized_prompts_pos_class.reshape(-1, d2)
        _, l2, d2 = self.tokenized_prompts_neg_class.shape
        tokenized_prompts_neg_class = self.tokenized_prompts_neg_class.reshape(-1, d2)
        tokenized_prompts_class = torch.cat((tokenized_prompts_pos_class, tokenized_prompts_neg_class), dim=0)

        _, l1, d1 = self.tokenized_prompts_pos_emo.shape
        tokenized_prompts_pos_emo = self.tokenized_prompts_pos_emo.reshape(-1, d1)
        _, l1, d1 = self.tokenized_prompts_neg_emo.shape
        tokenized_prompts_neg_emo = self.tokenized_prompts_neg_emo.reshape(-1, d1)
        tokenized_prompts_emo = torch.cat((tokenized_prompts_pos_emo, tokenized_prompts_neg_emo), dim=0)

        _, l, d = self.tokenized_prompts_pos.shape
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1, d)
        _, l, d = self.tokenized_prompts_neg.shape
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1, d)
        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg), dim=0)
        # torch.Size([1, 77]) torch.Size([1, 77]) torch.Size([2, 77])
        return prompts, tokenized_prompts, prompts_emo, tokenized_prompts_emo, prompts_class, tokenized_prompts_class, self.compound_prompts_text