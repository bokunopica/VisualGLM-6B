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


class PneumoniaClassifier(torch.nn.Module):
    def __init__(self, eva_args, **kwargs):
        super().__init__()
        self.vit = EVAViT(EVAViT.get_args(**eva_args))
        # self.mlp = nn.Linear()

    def forward(self, image, **kwargs):
        enc = self.vit(image)[0]
        print(enc.shape)
        return None
        # return self.glm_proj()
    
class CLFImageBaseProcessor():
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

class CLFImageEvalProcessor(CLFImageBaseProcessor):
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
