# Ben Kabongo
# September 2025


import argparse
import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def main(config):
    sts_df_path = os.path.join(config.sts_path)
    sts_df = pd.read_csv(sts_df_path, index_col=0)

    statements = sts_df['statement'].tolist()
    topics = sts_df['topic'].tolist()
    sentiments = sts_df['sentiment'].tolist()

    sentences = [f"{statement}. Topic: {topic} Sentiment: {sentiment}" 
                 for statement, topic, sentiment in zip(statements, topics, sentiments)]
    
    print(f"Number of sentences to embed: {len(sentences)}")
    print("Sample sentence:", sentences[0])

    model = SentenceTransformer(config.model_name)
    embeddings = model.encode(
        sentences, 
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=False,
        convert_to_tensor=True,
        batch_size=config.batch_size
    )

    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Type of embeddings: {type(embeddings)}")
    print(f"Norm of first embedding vector: {torch.norm(embeddings[0], p=2).item()}")

    embeddings_dir = os.path.join(config.output_dir, config.model_name)
    os.makedirs(embeddings_dir, exist_ok=True)

    torch.save(embeddings, os.path.join(embeddings_dir, "embeddings.pt"))
    print(f"Embeddings saved to {os.path.join(embeddings_dir, 'embeddings.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sts_path", 
        type=str, 
        required=True, 
        help="Directory containing the dataset."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Output directory."
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="all-MiniLM-L6-v2",
        help="Pre-trained model name."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Batch size for embedding generation."
    )
    
    args = parser.parse_args()
    main(args)