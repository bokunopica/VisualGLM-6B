import torch
import torch.nn as nn

from sat.model import ViTModel
from sat.model import BaseMixin
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


class PneumoniaClassifier(torch.nn.Module):
    def __init__(self, eva_args, **kwargs):
        super().__init__()
        self.vit = EVAViT(EVAViT.get_args(**eva_args))
        self.mlp_1 = nn.Linear(1408, 176)
        self.mlp_2 = nn.Linear(176, 16)
        
        self.mlp_3 = nn.Linear(4112, 4096)
        self.mlp_4 = nn.Linear(4096, 1024)
        self.mlp_5 = nn.Linear(1024, 2)
        self.activation = nn.ReLU()

    def forward(self, image, **kwargs):
        enc = self.vit(image)[0] # [4, 257, 1408]
        y = self.mlp_1(enc) # [4, 257, 176]
        y = self.mlp_2(y) # [4, 257, 16]
        y = torch.flatten(y, start_dim=1) # [4, 4112]
        y = self.mlp_3(y) # 4096
        y = self.mlp_4(y) # 1024
        y = self.mlp_5(y) # 2
        return self.activation(y)
        # return self.glm_proj()
    
# class CLFImageBaseProcessor():
#     def __init__(self, mean=None, std=None):
#         if mean is None:
#             mean = (0.48145466, 0.4578275, 0.40821073)
#         if std is None:
#             std = (0.26862954, 0.26130258, 0.27577711)

#         self.normalize = transforms.Normalize(mean, std)

# class CLFImageEvalProcessor(CLFImageBaseProcessor):
#     def __init__(self, image_size=384, mean=None, std=None):
#         super().__init__(mean=mean, std=std)

#         self.transform = transforms.Compose(
#             [
#                 transforms.Resize(
#                     (image_size, image_size), interpolation=InterpolationMode.BICUBIC
#                 ),
#                 transforms.ToTensor(),
#                 self.normalize,
#             ]
#         )

#     def __call__(self, item):
#         return self.transform(item)
