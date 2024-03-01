import json
from PIL import Image
from tqdm import trange
from torch import nn
from torch.utils.data import Dataset


def singleton(cls):
    instances = {}

    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return _singleton


@singleton
class FewShotDataset(Dataset):
    def __init__(self, path, processor, tokenizer, args):
        max_seq_length = args.max_source_length + args.max_target_length
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.images = []
        self.input_ids = []
        self.labels = []
        for item in data:
            image = processor(Image.open(item["img"]).convert("RGB"))
            input0 = tokenizer.encode("<img>", add_special_tokens=False)
            input1 = [tokenizer.pad_token_id] * args.image_length
            input2 = tokenizer.encode(
                "</img>问：" + item["prompt"] + "\n答：", add_special_tokens=False
            )
            a_ids = sum([input0, input1, input2], [])
            b_ids = tokenizer.encode(text=item["label"], add_special_tokens=False)
            if len(a_ids) > args.max_source_length - 1:
                a_ids = a_ids[: args.max_source_length - 1]
            if len(b_ids) > args.max_target_length - 2:
                b_ids = b_ids[: args.max_target_length - 2]
            pre_image = len(input0)
            input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

            context_length = input_ids.index(tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position + 1 :]

            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            if args.ignore_pad_token_for_loss:
                labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
            self.images.append(image)
            self.input_ids.append(input_ids)
            self.labels.append(labels)
        self.pre_image = pre_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "pre_image": self.pre_image,
        }


@singleton
class CovCTRDataset(Dataset):
    def __init__(self, path, processor, tokenizer, args):
        max_seq_length = args.max_source_length + args.max_target_length
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prefix_path = "/".join(path.split("/")[:-1])
        self.images = []
        self.input_ids = []
        self.labels = []
        self.covid_labels = []
        self.covid_one_hot_labels = []
        for i in trange(len(data)):
            item = data[i]
            img_filename = item["img"].split("/")[-1]
            image = processor(
                Image.open(f"{prefix_path}/{img_filename}").convert("RGB")
            )
            input0 = tokenizer.encode("<img>", add_special_tokens=False)
            input1 = [tokenizer.pad_token_id] * args.image_length
            ## cls_fusion
            # disease_prompt = ""
            # if "cls_fusion" in args:
            #     if item.get('is_covid', 0):
            #         disease_prompt = "该位受检者患有肺炎。"
            #     else:
            #         disease_prompt = "该位受检者未患有肺炎。"
            input2 = tokenizer.encode(
                "</img>问：" + item["prompt"] + "\n答：", add_special_tokens=False
            )
            a_ids = sum([input0, input1, input2], [])
            b_ids = tokenizer.encode(text=item["label"], add_special_tokens=False)
            if len(a_ids) > args.max_source_length - 1:
                a_ids = a_ids[: args.max_source_length - 1]
            if len(b_ids) > args.max_target_length - 2:
                b_ids = b_ids[: args.max_target_length - 2]
            pre_image = len(input0)
            input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

            context_length = input_ids.index(tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position + 1 :]

            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            if args.ignore_pad_token_for_loss:
                labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
            self.images.append(image)
            self.input_ids.append(input_ids)
            self.labels.append(labels)
            self.covid_labels.append([item["is_covid"]])
            self.covid_one_hot_labels.append(([0, 1] if item["is_covid"] else [1, 0]))
        self.pre_image = pre_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "pre_image": self.pre_image,
            "is_covid": self.covid_labels[idx],
            "is_covid_one_hot": self.covid_one_hot_labels[idx]
        }


@singleton
class MimicXrayDataset(Dataset):
    def __init__(self, path, processor, tokenizer, args):
        max_seq_length = args.max_source_length + args.max_target_length
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prefix_path = "/".join(path.split("/")[:-1])
        self.images = []
        self.input_ids = []
        self.labels = []
        for i in trange(len(data)):
            item = data[i]
            img_filename = item["file_name"]
            image = processor(
                Image.open(f"{prefix_path}/{img_filename}").convert("RGB")
            )
            input0 = tokenizer.encode("<img>", add_special_tokens=False)
            input1 = [tokenizer.pad_token_id] * args.image_length
            input2 = tokenizer.encode(
                "</img>问：" + item["prompt"] + "\n答：", add_special_tokens=False
            )
            a_ids = sum([input0, input1, input2], [])
            b_ids = tokenizer.encode(text=item["label"], add_special_tokens=False)
            if len(a_ids) > args.max_source_length - 1:
                a_ids = a_ids[: args.max_source_length - 1]
            if len(b_ids) > args.max_target_length - 2:
                b_ids = b_ids[: args.max_target_length - 2]
            pre_image = len(input0)
            input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

            context_length = input_ids.index(tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position + 1 :]

            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            if args.ignore_pad_token_for_loss:
                labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
            self.images.append(image)
            self.input_ids.append(input_ids)
            self.labels.append(labels)
        self.pre_image = pre_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "pre_image": self.pre_image,
        }
