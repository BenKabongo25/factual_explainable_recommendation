# Ben Kabongo
# October 2025


import ast
import math
import pandas as pd
import re
from tqdm import tqdm
from typing import Dict, List, Tuple
from utils.statement_utils import as_explanation_sentence, parse_list_field


def predicted_statement_to_explanation_doc(data_df: pd.DataFrame) -> pd.DataFrame:
    data = []
    for index, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Processing rows"):
        statement_triplets = parse_list_field(row["statements"])
        statement_ids = parse_list_field(row["statements_ids"])
        explanation = row["reference"]
        for id, triplet in zip(statement_ids, statement_triplets):
            statement = triplet["statement"]
            sentiment = triplet["sentiment"]
            formatted = as_explanation_sentence(statement, sentiment)
            topic = triplet["topic"]
            example = {
                "index": index,
                "statement_id": id,
                "statement": formatted,
                "topic": topic,
                "sentiment": sentiment,
                "document": explanation
            }
            data.append(example)
    return pd.DataFrame(data)


def ref_statement_to_explanation_gen_doc(
    ref_data_df: pd.DataFrame,
    pred_data_df: pd.DataFrame,
    sts_ref_df: pd.DataFrame
) -> pd.DataFrame:
    data = []
    for index, pred_row in tqdm(pred_data_df.iterrows(), total=len(pred_data_df), desc="Processing rows"):
        ref_row = ref_data_df.iloc[index]
        statement_ids = parse_list_field(ref_row["statement_ids"])
        explanation = pred_row["prediction"]
        for id in statement_ids:
            triplet = sts_ref_df[sts_ref_df["sid"] == id].iloc[0]
            if triplet.empty:
                continue
            statement = triplet["statement"]
            sentiment = triplet["sentiment"]
            formatted = as_explanation_sentence(statement, sentiment)
            topic = triplet["topic"]
            example = {
                "index": index,
                "statement_id": id,
                "statement": formatted,
                "topic": topic,
                "sentiment": sentiment,
                "document": explanation
            }
            data.append(example)
    return pd.DataFrame(data)


ITEM_METADATA_ID = "asin"
ITEM_METADATA_TITLE = "title"
ITEM_METADATA_DESCRIPTION = "description"

def approx_tokens(text: str) -> int:
    # approx ~4 chars / token
    return max(1, len(text) // 4)

def compute_num_chunks(text: str, target_tokens_per_chunk: int=1024, max_chunks: int=10) -> int:
    total_tokens = approx_tokens(text)
    n_chunks = max(1, min(max_chunks, math.ceil(total_tokens / target_tokens_per_chunk)))
    return n_chunks

def split_into_chunks(text: str, chunk_size: int=3) -> List[str]:
    # https://github.com/TanguyHsrt/seval-ex/blob/main/seval_package.py

    sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = [s.strip() for s in sentence_pattern.split(text) if s.strip()]

    # Ensure sentences end with appropriate punctuation
    sentences = [s + '.' if not s.endswith(('.', '!', '?')) else s for s in sentences]

    # Group sentences into chunks
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = '. '.join(sentences[i:i + chunk_size]) + '.'
        chunks.append(chunk)

    return chunks

def predicted_statement_to_item_doc(
    pred_data_df: pd.DataFrame,
    ref_data_df: pd.DataFrame,
    sts_ref_df: pd.DataFrame,
    data_dfs: List[pd.DataFrame],
    item_metadata_path: str,
    item_document_df: pd.DataFrame = None
) -> Tuple[pd.DataFrame]:
    if item_document_df is not None:
        item_documents = item_document_df["document"].to_dict()
    
    else:
        data_df = pd.concat(data_dfs, ignore_index=True)
        all_item_ids = data_df["item_id"].unique().tolist()
        item_data: Dict[str, Dict[str, str]] = {}
        with open(item_metadata_path, "r") as f:
            for line in tqdm(f, total=len(all_item_ids), desc="Processing item metadata"):
                item_metadata = ast.literal_eval(line.strip())

                if ITEM_METADATA_ID not in item_metadata:
                    continue

                item_id = item_metadata[ITEM_METADATA_ID]
                if item_id not in all_item_ids:
                    continue

                item_title = item_metadata.get(ITEM_METADATA_TITLE, None)
                if item_title:
                    item_title = (
                        item_title
                        .strip()
                        .lower()
                        .replace("\n", " ")
                    )

                item_description = item_metadata.get(ITEM_METADATA_DESCRIPTION, None)
                if item_description:
                    item_description = (
                        item_description
                        .strip()
                        .lower()
                        .replace("\n", " ")
                    )

                item_interactions_df = data_df[data_df["item_id"] == item_id]
                item_statement_ids = set()
                for _, interaction_row in item_interactions_df.iterrows():
                    statement_ids = parse_list_field(interaction_row["statement_ids"])
                    item_statement_ids.update(statement_ids)

                item_statements_df = sts_ref_df[sts_ref_df["sid"].isin(item_statement_ids)]
                item_statements = []
                for _, stmt_row in item_statements_df.iterrows():
                    statement = stmt_row["statement"]
                    sentiment = stmt_row["sentiment"]
                    formatted = as_explanation_sentence(statement, sentiment)
                    item_statements.append(formatted)

                item_data[item_id] = {
                    "title": item_title,
                    "description": item_description,
                    "statements": item_statements
                }

        item_documents = {}
        for item_id, item_data in item_data.items():
            statements_text = " ".join(item_data["statements"])
            document = f'TITLE: {str(item_data["title"])}\nDESCRIPTION: {str(item_data["description"])}\nSTATEMENTS: {statements_text}'
            item_documents[item_id] = document
        item_document_df = pd.DataFrame.from_dict(item_documents, orient='index', columns=['document'])

    data = []
    for index, pred_row in tqdm(pred_data_df.iterrows(), total=len(pred_data_df), desc="Processing rows"):
        statement_triplets = parse_list_field(pred_row["statements"])
        statement_ids = parse_list_field(pred_row["statements_ids"])
        ref_row = ref_data_df.iloc[index]
        item_id = ref_row["item_id"]
        item_document = item_documents.get(item_id, None)
        if item_document is None:
            continue
        for id, triplet in zip(statement_ids, statement_triplets):
            statement = triplet["statement"]
            sentiment = triplet["sentiment"]
            formatted = as_explanation_sentence(statement, sentiment)
            topic = triplet["topic"]
            chunk_size = compute_num_chunks(
                item_document, target_tokens_per_chunk=1024, max_chunks=10
            )
            if chunk_size > 1:
                chunks = split_into_chunks(item_document, chunk_size=chunk_size)
            else:
                chunks = [item_document]
            for i, chunk in enumerate(chunks):
                example = {
                    "index": index,
                    "statement_id": id,
                    "statement": formatted,
                    "topic": topic,
                    "sentiment": sentiment,
                    "item_id": item_id,
                    "chunk_id": i,
                    "document": chunk
                }
                data.append(example)
    input_data_df = pd.DataFrame(data)

    return input_data_df, item_document_df