# Ben Kabongo
# October 2025


import os
import pandas as pd

from evaluation.llm.create_files import (
    predicted_statement_to_explanation_doc, # statement2explanation
    ref_statement_to_explanation_gen_doc, # statement_ref2explanation_gen
    predicted_statement_to_item_doc, # statement2item_chunk
)
from utils.statement_utils import read_sts
from utils.text_generation_pipeline import get_parser, text_generation


SYSTEM_PROMPT = """You are a factual verifier. SINGLE TASK: decide whether a STATEMENT is fully supported by at least one passage in the DOCUMENT.

Decision rules (apply strictly):
1) Output "1" only if the DOCUMENT contains a passage that clearly ENTAILS the entire factual content of the STATEMENT (paraphrases/synonyms allowed; if the STATEMENT is negated, the negation must be explicit in the DOCUMENT).
2) If the STATEMENT has multiple sub-claims joined by “and”/commas, each sub-claim must be supported (sub-claims may be supported by different passages). If any sub-claim is unsupported, output "0".
3) Use NO knowledge outside the DOCUMENT. If evidence is missing, ambiguous, contradictory, or merely suggestive → output "0".
4) Numbers, quantities, dates, and named entities must match (obvious equivalences allowed, e.g., “dozen” = “12”).
5) Ignore off-topic content, opinions without factual content, and metadata without probative value.
6) If the DOCUMENT is empty or unreadable → output "0".

STRICT OUTPUT FORMAT:
- Respond with EXACTLY ONE character: "1" (supported) or "0" (not supported).
- No explanations, no extra text, no spaces, no quotes, no punctuation, and no additional newlines.
- Do not repeat the question or the statement.

Examples (NEVER reproduce in the output):
[Example 1] STATEMENT: "The phone has a 120 Hz display."
      DOCUMENT: "... OLED display with a 120 Hz refresh rate ..."
      OUTPUT: 1
[Example 2] STATEMENT: "The model weighs under 1 kg."
      DOCUMENT: "... weight: 1.2 kilograms ..."
      OUTPUT: 0
[Example 3] STATEMENT: "The battery is not removable."
      DOCUMENT: "... non-removable battery ..."
      OUTPUT: 1
[Example 4] STATEMENT: "Ships with a stylus and a case."
      DOCUMENT: "... ships with a stylus ..."
      OUTPUT: 0
"""


USER_PROMPT = """STATEMENT:
{stmt}

DOCUMENT:
{doc}

Answer now with 1 or 0 only.
OUTPUT:
"""

def format_message(example, system_role=True):
    statement = example["statement"]
    document = example["document"]

    example_prompt = USER_PROMPT.format(stmt=statement, doc=document)

    if system_role:
        example["messages"] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example_prompt}
        ]
    else:
        example["messages"] = [
            {"role": "user", "content": SYSTEM_PROMPT + "\n" + example_prompt}
        ]
    return example


def check_needs(config):
    _need_pred_data = config.task in [
        "statement2explanation",
        "statement_ref2explanation_gen",
        "statement2item_chunk",
    ]
    if _need_pred_data:
        assert config.pred_data_path is not None, "pred_data_path is required"

    _need_ref_data = config.task in [
        "statement_ref2explanation_gen",
        "statement2item_chunk",
    ]
    if _need_ref_data:
        assert config.ref_data_path is not None, "ref_data_path is required"

    _need_sts_ref = config.task in [
        "statement_ref2explanation_gen",
        "statement2item_chunk",
    ]
    if _need_sts_ref:
        assert config.sts_ref_path is not None, "sts_ref_path is required"

    _need_item_metadata = config.task == "statement2item_chunk"
    if _need_item_metadata:
        assert config.item_metadata_path, "item_metadata_path is required"


def main(config):
    check_needs(config)

    if config.task == "statement2explanation":
        print("Preparing data for statement2explanation task...")
        pred_data_df = pd.read_csv(config.pred_data_path)
        data_df = predicted_statement_to_explanation_doc(pred_data_df)

    elif config.task == "statement_ref2explanation_gen":
        print("Preparing data for statement_ref2explanation_gen task...")

        ref_data_df = pd.read_csv(config.ref_data_path)
        pred_data_df = pd.read_csv(config.pred_data_path)
        sts_ref_df = read_sts(config.sts_ref_path)

        data_df = ref_statement_to_explanation_gen_doc(
            ref_data_df, pred_data_df, sts_ref_df
        )

    elif config.task == "statement2item_chunk":
        print("Preparing data for statement2item_chunk task...")

        ref_data_df = pd.read_csv(config.ref_data_path)
        pred_data_df = pd.read_csv(config.pred_data_path)
        sts_ref_df = read_sts(config.sts_ref_path)

        data_dfs = []
        for split in ["train", "eval", "test"]:
            split_path = os.path.join(config.data_dir, f"{split}_data.csv")
            split_df = pd.read_csv(split_path)
            data_dfs.append(split_df)

        item_document_path = os.path.join(config.data_dir, "item_documents.csv")
        _item_document_df_already_exists = False
        item_document_df = None
        if os.path.exists(item_document_path):
            item_document_df = pd.read_csv(item_document_path, index_col=0)
            _item_document_df_already_exists = True

        data_df, item_document_df = predicted_statement_to_item_doc(
            pred_data_df,
            ref_data_df,
            sts_ref_df,
            data_dfs,
            config.item_metadata_path,
            item_document_df
        )

        if not _item_document_df_already_exists:
            item_document_df.to_csv(item_document_path, index=True)
            print(f"Saved item documents to {item_document_path}")

    else:
        raise ValueError(f"Unknown task: {config.task}")
    
    print(data_df.head())
    print(f"Total examples for {config.task}: {len(data_df)}")

    output_path = os.path.join(config.baseline_dir, f"{config.task}_labels.csv")
    state_dict_path = os.path.join(config.baseline_dir, f"{config.task}_state_dict.json")
    keeped_keys = data_df.columns.tolist() + ["label"]
    text_generation(
        config=config,
        data_df=data_df,
        format_message=lambda arg1, arg2, arg3: format_message(arg1, arg3),
        output_path=output_path,
        state_dict_path=state_dict_path,
        keeped_keys=keeped_keys,
        output_key="label",
        input_key="messages",
        use_prompt=False
    )


if __name__ == "__main__":
    parser = get_parser()

    parser.add_argument(
        "--task",
        type=str,
        default="statement2explanation",
        choices=["statement2explanation", "statement_ref2explanation_gen", "statement2item_chunk"],
        help="Task to perform"
    )

    parser.add_argument(
        "--baseline_dir",
        type=str,
        required=True,
        help="Path to baseline directory to save outputs"
    )
    parser.add_argument(
        "--sts_ref_path", 
        type=str, 
        default=None,
        help="Path to reference statements CSV"
    )
    parser.add_argument(
        "--sts_pred_path", 
        type=str, 
        default=None,
        help="Path to predicted statements CSV"
    )
    parser.add_argument(
        "--ref_data_path", 
        type=str, 
        required=True, 
        help="Path to ref_data CSV (per-review; contains reference SIDs)"
    )
    parser.add_argument(
        "--pred_data_path", 
        type=str, 
        required=True, 
        help="Path to pred_data CSV (per-review; contains predicted statements or SIDs)"
    )

    parser.add_argument(
        "--item_metadata_path", 
        type=str, 
        default=None,
        help="Path to item metadata file (required for statement2item_chunk task)"
    )

    config = parser.parse_args()
    main(config)
