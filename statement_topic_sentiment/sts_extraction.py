# Ben Kabongo
# September 2025


import argparse
import datasets
import json
import logging
import os
import pandas as pd
import re
import time
import torch
from transformers import pipeline


AMAZON_2014_ENTRIES = {
    "overall": "rating",
    "unixReviewTime": "timestamp",
    "reviewText": "review",
    "reviewerName": "user_name",
    "reviewerID": "user_id",
    "asin": "item_id",
    "summary": "review_title",
}


def format_message(example, prompt):
    review = example["review"]
    if len(review.split()) > config.max_length:
        review = " ".join(review.split()[:config.max_length])

    example["messages"] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Review: " + review + "\nOutput:"},
    ]
    return example


def extract_json_from_output(text):
    match = re.search(r"\[\s*{.*?}\s*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def empty_cache():
    with torch.no_grad():
        torch.cuda.empty_cache()


def main(config):
    logging.basicConfig(level=logging.INFO)

    os.makedirs(config.output_dir, exist_ok=True)
    path = os.path.join(config.output_dir, f"statement.csv")
    
    pipe = pipeline(
        "text-generation",
        model=config.model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        batch_size=config.batch_size,
    )

    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.tokenizer.padding_side = "left"
    terminators = [pipe.tokenizer.eos_token_id, pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    fmt = "json" if config.json_format else "csv"
    dataset = datasets.load_dataset(fmt, data_files={"train": config.dataset_path}, split="train", streaming=True)
    if os.path.exists(path) and config.skip_existing:
        logging.info(f"Skipping existing results in {path}")
        logging.info(f"Before skipping: {next(iter(dataset))}")
        df = pd.read_csv(path)
        length = len(df)
        dataset = dataset.skip(length)
        logging.info(f"After skipping: {next(iter(dataset))}")
    else:
        logging.info(f"No existing results found in {path}")
        
    if config.format == "amz14": # (Amazon Reviews 2014)
        for key, value in AMAZON_2014_ENTRIES.items():
            dataset = dataset.rename_column(key, value)

    prompt = open(config.prompt_text_file, "r").read().strip()
    dataset = dataset.map(lambda x: format_message(x, prompt))

    batch_dataset = dataset.batch(batch_size=config.batch_size)
    state_dict_dir = os.path.join(config.output_dir, "state_dicts")
    os.makedirs(state_dict_dir, exist_ok=True)
    state_dict_path = os.path.join(state_dict_dir, "state_dict.json")
    if os.path.exists(state_dict_path):
        with open(state_dict_path, "r") as f:
            state_dict = json.load(f)
        batch_dataset.load_state_dict(state_dict)
        logging.info(f"Loading state dict from {state_dict_path}")
    else:
        logging.info(f"No state dict found!")

    for batch in batch_dataset:
        start = time.time()
        outputs = pipe(
            batch["messages"],
            max_new_tokens=config.max_new_tokens,
            eos_token_id=terminators,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        end = time.time()
        logging.info(f"Batch processed in {end - start:.2f} seconds")
        raw_outputs = [out[0]["generated_text"][-1]["content"] for out in outputs]
        json_outputs = [extract_json_from_output(raw) for raw in raw_outputs]
         
        batch["statements"] = json_outputs
        del batch["messages"]
        df = pd.DataFrame(batch)
        df.to_csv(path, mode='a', header=not os.path.exists(path), index=False, escapechar='\\')

        state_dict = batch_dataset.state_dict()
        with open(state_dict_path, "w") as f:
            json.dump(state_dict, f)

        empty_cache()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str, 
        default="llama3/Meta-Llama-3-8B-Instruct", 
        help="Model path"
    )
    parser.add_argument(
        "--prompt_text_file", 
        type=str, 
        default="prompt.txt", 
        help="Path to the prompt text file"
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="dataset/restaurant.csv", 
        help="Dataset for statement/topic/sentiment extraction"
    )
    parser.add_argument(
        "--json_format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Specify if the dataset is in JSON format",
    )
    parser.add_argument(
        "--format", 
        type=str, 
        default="amz14", 
        choices=["amz14",], 
        help="Format of the dataset"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="dataset/", 
        help="Output path for results"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Batch size"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=768, 
        help="Maximum new tokens"
    )
    parser.add_argument(
        "--do_sample", 
        action=argparse.BooleanOptionalAction, 
        default=True, 
        help="Enable sampling"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.6, 
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9, 
        help="Top-p sampling probability"
    )
    parser.add_argument(
        "--skip_existing", 
        action=argparse.BooleanOptionalAction, 
        default=False, 
        help="Skip existing results"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=128, 
        help="Maximum length of the input sequence"
    )

    parser.set_defaults(json_format=True)
    parser.set_defaults(do_sample=True)
    parser.set_defaults(skip_existing=False)
    
    config = parser.parse_args()
    main(config)
