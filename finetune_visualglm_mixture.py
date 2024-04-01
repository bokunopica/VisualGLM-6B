import torch
import argparse
from sat import mpu, get_args, get_tokenizer
from sat.training.deepspeed_training import training_main
from model.blip2 import BlipImageEvalProcessor
from model.visualglm import FineTuneVisualGLMModel
from transformers import AutoTokenizer
from dataset import CovCTRDataset, MimicXrayDataset, FewShotDataset


def get_batch(data_iterator, args, timers):
    # Items and their type.
    
    # keys = ["input_ids", "labels"]
    keys = ["input_ids", "labels", "is_covid"]
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
    # data_c = mpu.broadcast_data(["is_covid"], data, torch.float32)
    # Unpack.
    tokens = data_b["input_ids"].long()
    labels = data_b["labels"].long()
    is_covid = data_b["is_covid"].long()
    img = data_i["image"]
    if args.fp16:
        img = img.half()
    return tokens, labels, img, data["pre_image"], is_covid


from torch.nn import CrossEntropyLoss


def forward_step(data_iterator, model, args, timers):
    """Forward step."""
    # Get the batch.
    timers("batch generator").start()
    tokens, labels, image, pre_image, is_covid = get_batch(data_iterator, args, timers)
    timers("batch generator").stop()

    logits = model(input_ids=tokens, image=image, pre_image=pre_image, is_covid=is_covid)[0]
    dtype = logits.dtype
    lm_logits = logits.to(torch.float32)

    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    lm_logits = lm_logits.to(dtype)
    loss = loss.to(dtype)
    return loss, {"loss": loss}


def create_dataset_function(path, args):
    # tokenizer = get_tokenizer(args)
    tokenizer = AutoTokenizer.from_pretrained("/home/qianq/model/chatglm-6b", trust_remote_code=True)
    image_processor = BlipImageEvalProcessor(224)
    dataset = CovCTRDataset(path, image_processor, tokenizer, args)
    # dataset = MimicXrayDataset(path, image_processor, tokenizer, args)
    return dataset


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

    model_type = "visualglm-6b"
    model, args = FineTuneVisualGLMModel.from_pretrained(
        model_type, args, build_only=True
    )
    for sub_model_name in model.mixins:
        print(sub_model_name)
        if sub_model_name in ["adapter", "ptuning", "lora"]:
            continue
        elif sub_model_name == "eva":
            model.mixins[sub_model_name].model.load_state_dict(
                torch.load(f"/home/qianq/mycodes/VisualGLM-6B/checkpoints/origin-qformer-cls-fusion/eva.model.ckpt"), 
                strict=False, # Fusion Model params 
            )
            for name, mod in model.mixins['eva'].model.named_children():
                if name.startswith('mlp_'):
                    # classifier mlp layers
                    mod.load_state_dict(
                        torch.load(f"/home/qianq/mycodes/VisualGLM-6B/checkpoints/clf-mlp/{name}.ckpt")
                    )
        else:
            model.mixins[sub_model_name].load_state_dict(
                torch.load(
                    f"/home/qianq/mycodes/VisualGLM-6B/checkpoints/origin/{sub_model_name}.ckpt"
                )
            )
    model.transformer.load_state_dict(
        torch.load(
            f"/home/qianq/mycodes/VisualGLM-6B/checkpoints/origin/chatglm-6b.ckpt"
        )
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    # tokenizer = get_tokenizer(args)
    tokenizer = AutoTokenizer.from_pretrained("/home/qianq/model/chatglm-6b", trust_remote_code=True)
    label_pad_token_id = (
        -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )

    def data_collator(examples):
        for example in examples:
            example["input_ids"] = torch.tensor(example["input_ids"], dtype=torch.long)
            example["labels"] = torch.tensor(example["labels"], dtype=torch.long)
            example["is_covid"] = torch.tensor(example["is_covid"], dtype=torch.long)
        ret = {
            "input_ids": torch.stack([example["input_ids"] for example in examples]),
            "labels": torch.stack([example["labels"] for example in examples]),
            "image": torch.stack([example["image"] for example in examples]),
            "pre_image": example["pre_image"],
            "is_covid": torch.stack([example["is_covid"] for example in examples]),
        }
        return ret

    training_main(
        args,
        model_cls=model,
        forward_step_function=forward_step,
        create_dataset_function=create_dataset_function,
        collate_fn=data_collator,
    )
