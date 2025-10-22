# Ben Kabongo
# August 2025


import ast
import os
import pandas as pd
import re

from utils.text_generation_pipeline import get_parser, text_generation


def format_message(example, prompt, system_role=True, num_words=200):
    prompt = prompt.replace("{N}", str(num_words))
    sample = example["item_prompt"]

    if system_role:
        example["messages"] = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "\nInput:" + sample + "\nOutput:\n"}
        ]
    else:
        example["messages"] = [
            {"role": "user", "content": prompt + "\nInput:" + sample + "\nOutput:\n"}
        ]
    return example


def extract_summary_text(text):
    try:
        pattern = r'"summarization"\s*:\s*"((?:[^"\\]|\\.)*)"'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            summary = match.group(1)
            summary = summary.replace('\\"', '"')
            summary = summary.replace('\\n', '\n')
            summary = summary.replace('\\t', '\t')
            summary = summary.replace('\\\\', '\\')
            return summary
        else:
            print(text)
            return None
    except Exception as e:
        print(f"Error with summary: {text}...")
        print(f"Error: {e}")
        raise ValueError("Could not parse summary text")


def main(config):
    train_df = pd.read_csv(os.path.join(config.data_dir, "topics", "train_data.csv"))
    train_df = train_df.sort_values(by="timestamp", ascending=True)
    train_item_ids = train_df["item_id"].unique().tolist()

    data = []
    with open(config.item_metadata_path, "r") as f:
        for line in f:
            item_metadata = ast.literal_eval(line.strip())

            if config.item_metadata_id not in item_metadata or config.item_metadata_title not in item_metadata:
                continue

            item_id = item_metadata[config.item_metadata_id]
            if item_id not in train_item_ids:
                continue

            item_title = item_metadata.get(config.item_metadata_title, None)
            if item_title:
                item_title = (
                    item_title
                    .strip()
                    .lower()
                    .replace("\n", " ")
                )
                if len(item_title.split()) > config.max_title_length:
                    item_title = " ".join(item_title.split()[:config.max_title_length])

            item_description = item_metadata.get("description", None)
            if item_description:
                item_description = (
                    item_description
                    .strip()
                    .lower()
                    .replace("\n", " ")
                )
                if len(item_description.split()) > config.max_description_length:
                    item_description = " ".join(item_description.split()[:config.max_description_length])

            #item_price = item_metadata.get("price", None)
            #item_categories = item_metadata.get("categories", None)
            #if len(item_categories) > 0:
            #    if isinstance(item_categories[0], list):
            #        item_categories = item_categories[0]
            #    item_categories = ", ".join(item_categories)

            item_prompt = 'BASIC INFORMATION: {"title": "' + str(item_title) + '",\n"description": "' + str(item_description) + '"} \n'

            reviews = train_df[train_df["item_id"] == item_id]["review"].tolist()
            if len(reviews) > config.num_interactions:
                reviews = reviews[-config.num_interactions:]

            for i in range(len(reviews)):
                review = (
                    reviews[i]
                    .strip()
                    .lower()
                    .replace("\n", " ")
                )
                if len(review.split()) > config.max_review_length:
                    review = " ".join(review.split()[:config.max_review_length])
                reviews[i] = review
            item_prompt += "USER FEEDBACK: " + str(reviews) + "\n"

            item = {
                "item_id": item_id,
                "title": item_title,
                "description": item_description,
                "item_prompt": item_prompt,
                #"price": item_price,
                #"categories": item_categories
            }
            data.append(item)

    data_df = pd.DataFrame(data)

    output_dir = os.path.join(config.output_dir, config.model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"item_summaries.csv")
    state_dict_path = os.path.join(output_dir, "item_state_dict.json")

    text_generation(
        config=config,
        data_df=data_df,
        format_message=lambda arg1, arg2, arg3: format_message(arg1, arg2, arg3, num_words=config.num_words),
        output_path=output_path,
        _dict_path=state_dict_path,
        keeped_keys=["item_id", "title", "description", "summary"],
        output_key="summary",
        input_key="messages"
    )

    df = pd.read_csv(output_path)
    df['summary_text'] = df['summary'].apply(extract_summary_text)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = get_parser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory to save the output files."
    )
    parser.add_argument(
        "--item_metadata_path",
        type=str,
        required=True,
        help="Path to the item metadata file."
    )
    parser.add_argument(
        "--item_metadata_id",
        type=str,
        default="asin",
        help="Key for item ID in the metadata file."
    )
    parser.add_argument(
        "--item_metadata_title",
        type=str,
        default="title",
        help="Key for item title in the metadata file."
    )
    parser.add_argument(
        "--num_words",
        type=int,
        default=100,
        help="Number of words for the summary."
    )
    parser.add_argument(
        "--max_title_length",
        type=int,
        default=64,
        help="Maximum length of the item title."
    )
    parser.add_argument(
        "--max_description_length",
        type=int,
        default=128,
        help="Maximum length of the item description."
    )
    parser.add_argument(
        "--num_interactions",
        type=int,
        default=5,
        help="Number of interactions to use for each item."
    )
    parser.add_argument(
        "--max_review_length",
        type=int,
        default=128,
        help="Maximum length of each review."
    )

    config = parser.parse_args()
    main(config)
