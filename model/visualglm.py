from torch import nn
import torch
from sat.model.official import ChatGLMModel
from sat.model.base_model import BaseMixin
from copy import deepcopy
import json
from .blip2 import BLIP2
from sat.model.finetune import PTuningV2Mixin
from sat.model.finetune.lora2 import LoraMixin
from sat.model.finetune import AdapterMixin
from sat.model.base_model import non_conflict
from sat.resources.urls import MODEL_URLS

MODEL_URLS['visualglm-6b'] = 'r2://visualglm-6b.zip'

# class LabelEmbedder(nn.Module):
#     def __init__(self, emb_dim=32, num_classes=2, act_name=("SWISH", {})):
#         super().__init__()
#         self.emb_dim = emb_dim
#         self.embedding = nn.Embedding(num_classes, emb_dim)

#     def forward(self, condition):
#         c = self.embedding(condition) #[B,] -> [B, C]
#         return c
    

class FusionModel(nn.Module):
    def __init__(self, emb_dim=4096, num_classes=2):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(num_classes, emb_dim)

    def forward(self, condition, image_emb):
        print('wwwwwwwwwwwwwwwwwww')
        cond_emb = self.embedding(condition)
        return cond_emb + image_emb


class ImageMixin(BaseMixin):
    def __init__(self, args):
        super().__init__()
        self.args = deepcopy(args)
        if hasattr(args, 'model_parallel_size'):
            args.eva_args['model_parallel_size'] = args.model_parallel_size
            args.qformer_args['model_parallel_size'] = args.model_parallel_size
        self.model = BLIP2(
            args.eva_args, 
            args.qformer_args,
        )
        if "cls_fusion" in args and args.cls_fusion:
            self.cls_fusion = False
            # TODO fusion model
            self.fusion_model = FusionModel(emb_dim=4096, num_classes=2)
            # TODO add classifier
            # self.classifier = 
        else:
            self.cls_fusion = False
            

    def word_embedding_forward(self, input_ids, output_cross_layer, **kw_args):
        if kw_args["pre_image"] > input_ids.shape[1] or kw_args.get("image", None) is None:
            return self.transformer.word_embeddings(input_ids)
        image_emb = self.model(**kw_args)
        # the image is inserted after 问：<img>, override 32 pads
        pre_id, pads, post_id = torch.tensor_split(input_ids, [kw_args["pre_image"], kw_args["pre_image"]+self.args.image_length], dim=1)
        pre_txt_emb = self.transformer.word_embeddings(pre_id)
        post_txt_emb = self.transformer.word_embeddings(post_id)
        
        if self.cls_fusion:
            # TODO classification tag fusion
            # TODO add classification embedding 疾病分类-COV-CTR
            condition = kw_args['is_covid']
            image_emb = self.fusion_model(condition, image_emb)
        return torch.cat([pre_txt_emb, image_emb, post_txt_emb], dim=1)


class VisualGLMModel(ChatGLMModel):
    def __init__(self, args, transformer=None, **kwargs):
        super().__init__(args, transformer=transformer, **kwargs)
        self.image_length = args.image_length
        self.add_mixin("eva", ImageMixin(args))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('VisualGLM', 'VisualGLM Configurations')
        group.add_argument('--image_length', type=int, default=32)
        group.add_argument('--eva_args', type=json.loads, default={})
        group.add_argument('--qformer_args', type=json.loads, default={})
        return super().add_model_specific_args(parser)
    

class AdjustAdapterMixin(AdapterMixin):
    @non_conflict
    def layer_forward(self, hidden_states, mask, old_impl, *args, **kw_args):
        """
        hidden_states: [batch, seq_len, hidden_size]
        mask: [(1, 1), seq_len, seq_len]
        """
        layer = self.transformer.layers[kw_args["layer_id"]]
        # Layer norm at the begining of the transformer layer.
        hidden_states = layer.input_layernorm(hidden_states)
        # Self attention.
        attention_output = layer.attention(hidden_states, mask, **kw_args)

        attention_output = attention_output + self.ff2[kw_args["layer_id"]](
            nn.functional.gelu(self.ff1[kw_args["layer_id"]](attention_output))
        )

        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = layer.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = layer.mlp(layernorm_output, **kw_args)
        mlp_output = mlp_output + self.ff4[kw_args["layer_id"]](
            nn.functional.gelu(self.ff3[kw_args["layer_id"]](mlp_output))
        )

        # Second residual connection.
        output = layernorm_output + mlp_output

        return output


class FineTuneVisualGLMModel(VisualGLMModel):
    def __init__(self, args, transformer=None, parallel_output=True, **kw_args):
        super().__init__(
            args, transformer=transformer, parallel_output=parallel_output, **kw_args
        )
        if args.use_ptuning:
            self.add_mixin(
                "ptuning",
                PTuningV2Mixin(
                    args.num_layers,
                    args.hidden_size // args.num_attention_heads,
                    args.num_attention_heads,
                    args.pre_seq_len,
                ),
            )
        if args.use_lora:
            self.add_mixin(
                "lora",
                LoraMixin(
                    args.num_layers,
                    args.lora_rank,
                    layer_range=args.layer_range,
                ),
                reinit=True,
            )
            # self.get_mixin("eva").model.glm_proj = replace_linear_with_lora(self.get_mixin("eva").model.glm_proj, LoraLinear, args.lora_rank)
        elif args.use_qlora:
            self.add_mixin(
                "lora",
                LoraMixin(
                    args.num_layers,
                    args.lora_rank,
                    layer_range=args.layer_range,
                    qlora=True,
                ),
                reinit=True,
            )
        elif args.use_adapter:
            # adapter finetune
            self.add_mixin(
                "adapter",
                AdjustAdapterMixin(
                    num_layers=args.num_layers, # 28-transformer一致
                    hidden_size=args.hidden_size, # 4096
                    adapter_hidden=args.adapter_hidden, # specified in .sh
                ),
            )
            pass
        self.args = args

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group(
            "VisualGLM-finetune", "VisualGLM finetune Configurations"
        )
        group.add_argument("--pre_seq_len", type=int, default=8)
        group.add_argument("--lora_rank", type=int, default=10)
        group.add_argument("--use_ptuning", action="store_true")
        group.add_argument("--use_lora", action="store_true")
        group.add_argument("--use_qlora", action="store_true")
        group.add_argument("--layer_range", nargs="+", type=int, default=None)
        group.add_argument("--use_adapter", action="store_true")
        group.add_argument("--adapter_hidden", type=int, default=128)
        group.add_argument("--adapter_num_layers", type=int, default=28)
        group.add_argument("--use_freeze", action="store_true")
        group.add_argument("--unfreeze_layers", type=str, default="")
        group.add_argument("--train_qformer", action="store_true")
        group.add_argument("--train_vit_transformer", type=str, default="")
        # group.add_argument("--use_classification_info", action="store_true")
        
        return super().add_model_specific_args(parser)

    def disable_untrainable_params(self):
        enable = []
        if self.args.use_ptuning:
            enable.extend(["ptuning"])
        if self.args.use_lora or self.args.use_qlora:
            enable.extend(["matrix_A", "matrix_B"])
        if self.args.use_freeze:
            unfreeze_layers = self.args.unfreeze_layers.split(',')
        else:
            unfreeze_layers = []

        if self.args.train_vit_transformer:
            train_vit_transformers = self.args.train_vit_transformer.split(',')
        else:
            train_vit_transformers = []


        print('------------unfreeze_layer--------------')
        for n, p in self.named_parameters():
            flag = False
            # adapter unfreeze
            if self.args.use_adapter and n.startswith("mixins.adapter"):
                flag = True
            elif self.args.use_freeze:
                for unfreeze_layer in unfreeze_layers:
                    if n.startswith(f"transformer.layers.{unfreeze_layer}."):
                        flag = True
                        break
            elif self.args.train_qformer and n.startswith("mixins.eva.model.qformer"):
                # qformer unfreeze
                flag = True
            elif self.args.train_vit_transformer and n.startswith("mixins.eva.model.vit.transformer.layers."):
                # Vit unfreeze
                _n = n.replace("mixins.eva.model.vit.transformer.layers.", "")
                _n = _n.split('.')[0]
                if _n in train_vit_transformers:
                    flag = True
            else:
                for e in enable:
                    if e.lower() in n.lower():
                        flag = True
                        break
            if not flag:
                p.requires_grad_(False)
            else:
                print(n)
        print('------------unfreeze_layer--------------')
