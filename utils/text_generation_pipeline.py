# Ben Kabongo
# August 2025


import argparse
import ast
import datasets
import json
import logging
import os
import pandas as pd
import time
import torch
from transformers import pipeline
from typing import Callable, List


def empty_cache():
    with torch.no_grad():
        torch.cuda.empty_cache()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3/Meta-Llama-3.1-8B-Instruct", help="Model path")
    parser.add_argument("--model_name", type=str, default="llama3.1-8b-it", help="Name of the model for output directory")
    parser.add_argument("--prompt_text_file", type=str, default="prompt.txt", help="Path to the prompt text file")
    parser.add_argument("--data_dir", type=str, default="dataset/", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum new tokens")
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True, help="Enable sampling")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling probability")
    return parser


def text_generation(
    config,
    data_df: pd.DataFrame,
    format_message: Callable,
    output_path: str,
    state_dict_path: str,
    keeped_keys: List[str],
    output_key: str,
    input_key: str = "messages",
    use_prompt: bool = True,
):
    logging.basicConfig(level=logging.INFO)

    pipe = pipeline(
        "text-generation",
        model=config.model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        batch_size=config.batch_size,
    )

    system_role = True
    if "gemma-7b-it" in config.model_name.lower():
        system_role = False
        terminators = []

    if "llama" in config.model_name.lower():
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        pipe.tokenizer.padding_side = "left"
        terminators = [pipe.tokenizer.eos_token_id, pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    prompt = ""
    if use_prompt:
        prompt = open(config.prompt_text_file, "r").read().strip()

    dataset = datasets.Dataset.from_pandas(data_df, split="train")
    dataset = dataset.map(lambda x: format_message(x, prompt, system_role))
    iterable_dataset = dataset.to_iterable_dataset()
    batch_dataset = iterable_dataset.batch(batch_size=config.batch_size)

    if os.path.exists(state_dict_path):
        with open(state_dict_path, "r") as f:
            state_dict = json.load(f)
        batch_dataset.load_state_dict(state_dict)
        logging.info(f"Loading state dict from {state_dict_path}")
    else:
        logging.info(f"No state dict found!")

    if output_key not in keeped_keys:
        keeped_keys.append(output_key)

    for batch_idx, batch in enumerate(batch_dataset):
        start = time.time()
        outputs = pipe(
            batch[input_key],
            max_new_tokens=config.max_new_tokens,
            eos_token_id=terminators,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
            batch_size=config.batch_size,
        )
        end = time.time()
        logging.info(f"Batch processed in {end - start:.2f} seconds")
        outputs = [out[0]["generated_text"][-1]["content"].strip() for out in outputs]
         
        batch[output_key] = outputs
        batch = {key: batch[key] for key in keeped_keys if key in batch}
        df = pd.DataFrame(batch)
        df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False, escapechar='\\')

        state_dict = batch_dataset.state_dict()
        with open(state_dict_path, "w") as f:
            json.dump(state_dict, f)

        if batch_idx == 0:
            for key in batch.keys():
                logging.info(f"{key}: {batch[key][:5]}")

        empty_cache()
        