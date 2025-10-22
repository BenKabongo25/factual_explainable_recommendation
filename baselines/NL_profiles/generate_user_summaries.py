# Ben Kabongo
# August 2025


import ast
import os
import pandas as pd
import re

from utils.text_generation_pipeline import get_parser, text_generation


def format_message(example, prompt, system_role=True, num_words=200):
    prompt = prompt.replace("{N}", str(num_words))
    sample = example["user_prompt"]

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
    train_user_ids = train_df["user_id"].unique().tolist()

    output_dir = os.path.join(config.output_dir, config.model_name)
    os.makedirs(output_dir, exist_ok=True)

    item_summaries_df = pd.read_csv(os.path.join(output_dir, "item_summaries.csv"))

    data = []
    for user_id in train_user_ids:
        user_interactions_df = train_df[train_df["user_id"] == user_id]
        
        user_prompt = "PURCHASED PRODUCTS: ["
        cpt = 0
        for idx in range(len(user_interactions_df) -1, -1, -1):
            row = user_interactions_df.iloc[idx]

            item_id = row["item_id"]
            item_data = item_summaries_df[item_summaries_df["item_id"] == item_id]
            if item_data.empty:
                continue

            item_title = str(item_data["title"].values[0])
            if len(item_title.split()) > config.max_title_length:
                item_title = " ".join(item_title.split()[:config.max_title_length])

            item_description = item_data["summary_text"].values[0]
            if not isinstance(item_description, str):
                #continue
                item_description = item_data["description"].values[0]
            item_description = str(item_description)

            if len(item_description.split()) > config.max_description_length:
                item_description = " ".join(item_description.split()[:config.max_description_length])

            review = str(row["review"])
            if len(review.split()) > config.max_review_length:
                review = " ".join(review.split()[:config.max_review_length])

            user_prompt += (
                '{"title": "' + str(item_title) + 
                '",\n"description": "' + str(item_description) + 
                '",\n"review": "' + str(item_description) + '"} \n'
            )

            cpt += 1
            if cpt > config.num_interactions:
                break

        user_prompt += "]"
        user = {
            "user_id": user_id,
            "user_prompt": user_prompt,
        }
        data.append(user)

    data_df = pd.DataFrame(data)
    output_path = os.path.join(output_dir, f"user_summaries.csv")
    state_dict_path = os.path.join(output_dir, "user_state_dict.json")

    print(data_df.head())

    text_generation(
        config=config,
        data_df=data_df,
        format_message=lambda arg1, arg2, arg3: format_message(arg1, arg2, arg3, num_words=config.num_words),
        output_path=output_path,
        state_dict_path=state_dict_path,
        keeped_keys=["user_id", "summary"],
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
