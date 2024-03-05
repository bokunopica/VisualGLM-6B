import torch
import torch.nn as nn

from sat.model import ViTModel, BaseModel
from sat.model import BaseMixin
from sat import AutoModel
from copy import deepcopy
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

class LNFinalyMixin(BaseMixin):
    def __init__(self, hidden_size):
        super().__init__()
        self.ln_vision = nn.LayerNorm(hidden_size)

    def final_forward(self, logits, **kw_args):
        return self.ln_vision(logits)

class EVAViT(ViTModel):
    def __init__(self, args, transformer=None, parallel_output=True, **kwargs):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, **kwargs)
        self.del_mixin("cls")
        self.add_mixin("cls", LNFinalyMixin(args.hidden_size))
    
    def forward(self, image):
        batch_size = image.size(0)
        input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=image.device)
        attention_mask = torch.tensor([[1.]], dtype=image.dtype, device=image.device)
        return super().forward(input_ids=input_ids, position_ids=None, attention_mask=attention_mask, image=image)

class QFormer(BaseModel):
    def __init__(self, args, transformer=None, parallel_output=True, **kwargs):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, activation_func=nn.functional.gelu, **kwargs)
        self.transformer.position_embeddings = None
    
    def final_forward(self, logits, **kw_args):
        return logits

    def position_embedding_forward(self, position_ids, **kw_args):
        return None
    
    def forward(self, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        input_ids = torch.arange(32, dtype=torch.long, device=encoder_outputs.device).unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.tensor([[1.]], dtype=encoder_outputs.dtype, device=encoder_outputs.device)
        cross_attention_mask = torch.tensor([[1.]], dtype=encoder_outputs.dtype, device=encoder_outputs.device)
        return super().forward(input_ids=input_ids, position_ids=None, attention_mask=attention_mask, encoder_outputs=encoder_outputs, cross_attention_mask=cross_attention_mask)


class FusionModel(nn.Module):
    def __init__(self, cond_emb_dim=8, image_emb_dim=768, num_classes=2, nhead=8, proj_output_dim=768):
        super().__init__()
        self.cond_emb_dim = cond_emb_dim
        self.image_emb_dim = image_emb_dim
        self.emb_dim = self.cond_emb_dim + self.image_emb_dim
        self.cond_embedding_model = nn.Embedding(num_classes, self.cond_emb_dim*32) # 对应32个queries
        self.transformer_encoder = nn.TransformerEncoderLayer(
            self.cond_emb_dim + self.image_emb_dim, 
            nhead, 
            self.emb_dim, 
            activation='relu'
        )
        self.proj_output_dim = proj_output_dim
        self.proj = nn.Linear(self.emb_dim, self.proj_output_dim)

    def forward(self, condition, image_emb):
        cond_emb = self.cond_embedding_model(condition)
        img_emb_shape = image_emb.shape
        cond_emb =cond_emb.reshape((img_emb_shape[0], img_emb_shape[1], -1))
        emb = torch.concat([cond_emb, image_emb], dim=2)
        tf_emb = self.transformer_encoder(emb)
        out = self.proj(tf_emb)
        return out

class BLIP2(torch.nn.Module):
    def __init__(self, eva_args, qformer_args, vit=None, qformer=None, cls_fusion=None, **kwargs):
        super().__init__()
        if vit is not None:
            self.vit = vit
        else:
            self.vit = EVAViT(EVAViT.get_args(**eva_args))
        if qformer is not None:
            self.qformer = qformer
        else:
            self.qformer = QFormer(QFormer.get_args(**qformer_args))

        if cls_fusion:
            self.mlp_1 = nn.Linear(1408, 176)
            self.mlp_2 = nn.Linear(176, 16)
            self.mlp_3 = nn.Linear(4112, 4096)
            self.mlp_4 = nn.Linear(4096, 1024)
            self.mlp_5 = nn.Linear(1024, 2)
            self.activation = nn.ReLU()
            self.fusion_model = FusionModel(
                cond_emb_dim=8,
                image_emb_dim=768,
                num_classes=2,
                nhead=8,
                proj_output_dim=768
            )
        else:
            self.fusion_model = None
        
        self.glm_proj = nn.Linear(768, 4096).to(self.qformer.parameters().__next__().device).to(self.qformer.parameters().__next__().dtype)

    def forward(self, image, **kwargs):
        enc = self.vit(image)[0]
        out = self.qformer(enc)[0]
        if self.fusion_model:
            # cls_fusion
            is_covid = self.clf_forward(enc)
            out = self.fusion_model(condition = is_covid, image_emb=out)
        # print(enc.shape) [1, 257, 1408]
        # print(out.shape) [1, 32, 768]
        # print(res.shape) [1, 32, 4096]
        return self.glm_proj(out)
    
    def clf_forward(self, enc):
        y = self.mlp_1(enc) # [4, 257, 176]
        y = self.mlp_2(y) # [4, 257, 16]
        y = torch.flatten(y, start_dim=1) # [4, 4112]
        y = self.mlp_3(y) # 4096
        y = self.mlp_4(y) # 1024
        y = self.mlp_5(y) # 2
        y = self.activation(y)
        res = y.argmax(dim=1)
        print('------------------')
        print(res)
        print('------------------')
        return res
    
    
class BlipImageBaseProcessor():
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

class BlipImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=384, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)
