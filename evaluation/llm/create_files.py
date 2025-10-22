# Ben Kabongo
# October 2025


import pandas as pd
from tqdm import tqdm
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

