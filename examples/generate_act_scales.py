import sys
sys.path.append("..")
import torch
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse

from smoothquant.calibration import get_act_scales

def build_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    kwargs = {"torch_dtype": torch.float16, "device_map": "cuda:0"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    """
    print('Adding special tokens.')

    print(f'!!! E-size before: {model.resize_token_embeddings(len(tokenizer)).weight.shape}')
    tokenizer.add_special_tokens({
            "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
            "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
            "unk_token": tokenizer.convert_ids_to_tokens(
                model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
            ),
    })

    print(f'!!! Tokens: {model.config.eos_token_id} {model.config.bos_token_id} {model.config.pad_token_id} {tokenizer.pad_token_id}')

    print('!!!!!!! MAYBE WTF HERE !!!!!!!')

    model.resize_token_embeddings(len(tokenizer))
    print(f'!!! E-size after: {model.resize_token_embeddings(len(tokenizer)).weight.shape}')
    """

    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default='facebook/opt-1.3b', help='model name')
    parser.add_argument('--output-path', type=str, default='act_scales/opt-1.3b.pt',
                        help='where to save the act scales')
    parser.add_argument('--dataset-path', type=str, default='dataset/val.jsonl.zst',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    parser.add_argument('--num-samples', type=int, default=512)
    parser.add_argument('--seq-len', type=int, default=512)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model_name)

#    if not os.path.exists(args.dataset_path):
#        print(f'Cannot find the dataset at {args.dataset_path}')
#        print('Please download the Pile dataset and put the validation set at the path')
#        print('You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst')
#        raise FileNotFoundError

    act_scales = get_act_scales(model, tokenizer, args.dataset_path,
                                args.num_samples, args.seq_len)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)


if __name__ == '__main__':
    main()
