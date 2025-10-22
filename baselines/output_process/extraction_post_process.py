# Ben Kabongo
# October 2025



import argparse
import ast
import pandas as pd
from baselines.output_process.topic_mapping_utils import TOPIC_MAPPING


def process_dataset(data_df, corrected_topics):
    corrected_topics_list = list(corrected_topics)

    all_triplets = {}
    cleaned_statements = []
    cleaned_statements_ids = []
    
    for index, triplet_list in enumerate(data_df["statements"].tolist(), start=1):
        try:
            triplet_list = ast.literal_eval(triplet_list)
        except:
            cleaned_statements.append(None)
            cleaned_statements_ids.append(None)
            continue

        new_triplet_list = []
        new_triplet_ids_list = []
        
        for triplet in triplet_list:
            if not triplet: continue

            topic = triplet.get("topic")
            if topic: topic = str(topic).lower().replace("-", " ").replace("_", " ").strip()
            #else: continue
            #if topic not in (topics + corrected_topics_list): continue
            if topic in corrected_topics_list: topic = corrected_topics[topic]

            sentiment = triplet.get("sentiment")
            if sentiment: sentiment = str(sentiment).lower().strip()
            #if sentiment not in sentiment_map: continue
                
            statement = triplet.get("statement")
            if statement: statement = str(statement).lower().replace("-", " ").replace("_", " ").strip()
            else: continue

            triplet_tuple = (statement, topic, sentiment)
            if triplet_tuple not in all_triplets:
                new_id = len(all_triplets)
                all_triplets[triplet_tuple] = {}
                all_triplets[triplet_tuple]["id"] = new_id
                all_triplets[triplet_tuple]["freq"] = 0
                
            all_triplets[triplet_tuple]["freq"] += 1
            new_triplet = {"statement": statement, "topic": topic, "sentiment": sentiment}

            new_triplet_list.append(new_triplet)
            s_id = all_triplets[triplet_tuple]["id"]
            new_triplet_ids_list.append(s_id)

        if len(new_triplet_list) == 0:
            new_triplet_list = []
            new_triplet_ids_list = []
            
        cleaned_statements.append(new_triplet_list)
        cleaned_statements_ids.append(new_triplet_ids_list)

        if index % 10_000 == 0:
            print("10000 samples processed...")

    print("Done!")
    n_none = cleaned_statements.count(None)
    print("Number of fails:", n_none)
    print("% of fails:", n_none/len(data_df))

    new_data_df = pd.DataFrame(data_df)
    new_data_df["statements"] = cleaned_statements
    new_data_df["statements_ids"] = cleaned_statements_ids
        
    columns = ["reference", "prediction", "statements", "statements_ids"]
    #new_data_df = new_data_df[columns]
    #new_data_df = new_data_df.dropna(subset=columns)

    all_triplets_data = []
    for (statement, topic, sentiment), info in all_triplets.items():
        triplet_data = {
            "id": info["id"],
            "statement": statement,
            "topic": topic,
            "sentiment": sentiment,
            "frequency": info["freq"]
        }
        all_triplets_data.append(triplet_data)

    all_triplets_df = pd.DataFrame(all_triplets_data)

    return new_data_df, all_triplets_df


def main(config):
    print("Loading data...")
    data_df = pd.read_csv(config.input_path)
    print(f"Input data:\n{data_df.head()}")
    print(f"Number of samples in input data: {len(data_df)}")
    print("Data loading completed.\n")

    print(f"Processing dataset {config.dataset_name}...")
    corrected_topics = TOPIC_MAPPING.get(config.dataset_name, {})
    processed_data_df, all_triplets_df = process_dataset(data_df, corrected_topics)
    print(f"Processed data:\n{processed_data_df.head()}")
    print(f"Number of samples in processed data: {len(processed_data_df)}")
    print(f"All triplets data:\n{all_triplets_df.head()}")
    print(f"Number of unique triplets: {len(all_triplets_df)}")
    print("Dataset processing completed.\n")

    print("Saving processed data...")
    processed_data_df.to_csv(config.output_data_path, index=False)
    all_triplets_df.to_csv(config.output_triplets_path, index=False)
    print(f"Processed data saved to: {config.output_data_path}")
    print(f"All triplets data saved to: {config.output_triplets_path}")
    print("Data saving completed.\n")

    print("All tasks completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process extracted review statements.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input CSV file containing extracted statements.")
    parser.add_argument("--output_data_path", type=str, required=True, help="Path to save the processed data CSV file.")
    parser.add_argument("--output_triplets_path", type=str, required=True, help="Path to save the unique triplets CSV file.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (e.g., 'Toys', 'Clothes').")

    args = parser.parse_args()
    main(args)
