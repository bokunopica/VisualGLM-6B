import torch
import argparse
from sat import mpu, get_args, get_tokenizer
from sat.training.deepspeed_training import training_main
from model.visualglm import FineTuneVisualGLMModel
from transformers import AutoTokenizer
from dataset import CovCTRDataset

from model.vit_classifier import PneumoniaClassifier
from model.blip2 import BlipImageEvalProcessor
from tqdm import trange
import torch.nn.functional as F



def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ["is_covid_one_hot"]
    datatype = torch.int64
    # Broadcast data.
    timers("data loader").start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers("data loader").stop()
    data_b = mpu.broadcast_data(keys, data, datatype)
    data_i = mpu.broadcast_data(["image"], data, torch.float32)
    # Unpack.
    # tokens = data_b["input_ids"].long()
    # labels = data_b["labels"].long()
    is_covid_one_hot = data_b["is_covid_one_hot"].long()
    # is_covid = torch.nn.functional.one_hot(is_covid, num_classes=2)
    img = data_i["image"]
    if args.fp16:
        img = img.half()
    return img, is_covid_one_hot


from torch.nn import CrossEntropyLoss


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers("batch generator").start()
    image, is_covid_one_hot = get_batch(data_iterator, args, timers)
    
    timers("batch generator").stop()
    logits = model(image=image)
    # dtype = logits.dtype
    # lm_logits = logits.to(torch.float32)

    # # Shift so that tokens < n predict n
    # shift_logits = lm_logits[..., :-1, :].contiguous()
    # shift_labels = labels[..., 1:].contiguous()
    
    # # Flatten the tokens
    # loss_fct = CrossEntropyLoss()
    # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # lm_logits = lm_logits.to(dtype)
    # loss = loss.to(dtype)
    dtype = logits.dtype
    cmp_logits = logits.to(torch.float32)
    is_covid_one_hot = is_covid_one_hot.to(torch.float32)
    
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(cmp_logits, is_covid_one_hot)

    cmp_logits = cmp_logits.to(dtype)
    loss = loss.to(dtype)
    return loss, {"loss": loss}



if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument("--max_source_length", type=int)
    py_parser.add_argument("--max_target_length", type=int)
    py_parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True)
    # py_parser.add_argument('--old_checkpoint', action="store_true")
    py_parser.add_argument("--source_prefix", type=str, default="")
    py_parser = FineTuneVisualGLMModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    args.device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained("/home/qianq/model/chatglm-6b", trust_remote_code=True)
    image_processor = BlipImageEvalProcessor(224)
    path = '/home/qianq/data/COV-CTR/eval.json'
    dataset = CovCTRDataset(path, image_processor, tokenizer, args)
    

    model = PneumoniaClassifier(
        eva_args={
            "num_layers": 39,
            "hidden_size": 1408,
            "num_attention_heads": 16,
            "vocab_size": 1,
            "layernorm_order": "pre",
            "model_parallel_size": 1,
            "max_sequence_length": 257,
            "inner_hidden_size": 6144,
            "use_final_layernorm": False,
            "layernorm_epsilon": 1e-06,
            "image_size": [224, 224],
            "pre_len": 1,
            "post_len": 0,
            "in_channels": 3,
            "num_classes": 0,
            "patch_size": 14,
        }
    )
    # vit冻结
    model.vit.requires_grad_(False)

    if torch.cuda.is_available():
        model = model.to("cuda")

    def data_collator(examples):
        for example in examples:
            example["is_covid_one_hot"] = torch.tensor(example["is_covid_one_hot"], dtype=torch.long)
        ret = {
            "image": torch.stack([example["image"] for example in examples]),
            "is_covid_one_hot": torch.stack([example["is_covid_one_hot"] for example in examples]),
        }
        return ret

    training_main(
        args,
        model_cls=model,
        forward_step_function=forward_step,
        collate_fn=data_collator,
    )
