# Ben Kabongo
# October 2025


import pandas as pd
from tqdm import tqdm
from typing import List
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


def predicted_statement_to_item_doc(
    pred_data_df: pd.DataFrame,
    data_dfs: List[pd.DataFrame],
    explanation_dfs: List[pd.DataFrame],
) -> pd.DataFrame:
    