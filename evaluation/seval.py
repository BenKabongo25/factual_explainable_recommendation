# https://github.com/TanguyHsrt/seval-ex/blob/main/seval_package.py

import argparse
import json
from typing import Dict, List, Optional
import ollama
from datasets import load_dataset
from tqdm import tqdm

#### USAGE ####
#--text "Albert Einstein was born in Ulm, Germany in 1879. He developed the theory of relativity and won the Nobel Prize in Physics in 1921." --model "qwen:7b"   --summary "Einstein was born in Germany and won a Nobel Prize."
# --model "qwen:7b" --dataset "mteb/summeval"

class SEvalPipeline:
    """
        Main pipeline for SEval-Ex summary evaluation.

        This class implements the two-stage pipeline described in the paper:
        1. Statement Extraction - Decomposing text into atomic statements
        2. Verdict Reasoning - Matching and classifying statements

        The pipeline supports different evaluation methods including:
        - StSum_Text: Direct matching of summary statements to source text
        - Base: Processing the entire text as a single unit
        - Chunked: Segmenting the text into chunks for better context preservation
        """
    def __init__(self, model_name: str = "qwen2.5:72b", method: str = "StSum_Text", chunk_size: int = 3,
                 max_text_length: int = 500):
        """
        Initialize the SEval-Ex pipeline with model and method settings.

        Args:
            model_name (str): Name of the LLM model to use for statement extraction and verification
            method (str): Evaluation method to use:
                         - "StSum_Text": Direct matching of summary statements to source text -> the best method
                         - "Base": Processing entire text as a single unit
                         - "Chunked": Segmenting text into smaller chunks
            chunk_size (int): Number of sentences per chunk for preserving local context
            max_text_length (int): Maximum word count before chunking is applied
        """
        self.model_name = model_name
        self.method = method
        self.chunk_size = chunk_size
        self.max_text_length = max_text_length
        self.temperature = 0.0001

    def split_into_chunks(self, text: str) -> List[str]:
        """
            Split text into semantically coherent chunks to preserve local context.

            As described in Section 3.2 of the paper, processing text in smaller chunks
            helps to maintain contextual relationships between sentences and improves
            statement extraction accuracy.

            Args:
                text (str): The source text to be split into chunks

            Returns:
                List[str]: A list of text chunks, each containing self.chunk_size sentences
            """

        # Split text into sentences (simple implementation - can be enhanced)
        #sentences = [s.strip() for s in text.split('.') if s.strip()]

        # More robust sentence splitting using regex
        import re
        sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = [s.strip() for s in sentence_pattern.split(text) if s.strip()]

        # Ensure sentences end with appropriate punctuation
        sentences = [s + '.' if not s.endswith(('.', '!', '?')) else s for s in sentences]

        # Group sentences into chunks
        chunks = []
        for i in range(0, len(sentences), self.chunk_size):
            chunk = '. '.join(sentences[i:i + self.chunk_size]) + '.'
            chunks.append(chunk)

        return chunks

    def extract_statements(self, text: str) -> List[str]:
        """
            Extract atomic statements from text using LLM.

            Implements the Statement Extraction phase from the paper's pipeline.
            An atomic statement is a self-contained unit of information that represents
            a single fact or claim.

            Args:
                text (str): The source text from which to extract statements

            Returns:
                List[str]: A list of extracted atomic statements

            Note:
                When the "Chunked" method is used, the text is first divided into
                smaller chunks to preserve local context before statement extraction.
            """

        prompt = """
        Extract key statements from the following text. Each statement should be a single, self-contained fact or claim. Do not add any ponctuation or words. Each statement should be on a new line.
        
        Here an example of a statement extraction: 
        Text: 
        "Albert Einstein was born in Barcelona Spain, 1879" 
        Extracted statements:
        "Albert Einstein was born in Spain",
        "Albert Einstein was born in Barcelona",
        "Albert Einstein was born in 1879"
        
        
        Text: {text}
        
        Extracted statements:
        """.format(text=text)

        if self.method == "Chunked":
            chunks = self.split_into_chunks(text)
            all_statements = []

            for chunk in chunks:
                chunk_prompt = prompt.format(text=chunk)
                response = ollama.generate(
                    model=self.model_name,
                    prompt=chunk_prompt,
                    options={"temperature": self.temperature}
                )
                statements = [s.strip() for s in response['response'].split('\n') if s.strip()]
                all_statements.extend(statements)

            return all_statements
        else:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt.format(text=text),
                options={"temperature": self.temperature}
            )

            statements = [s.strip() for s in response['response'].split('\n') if s.strip()]
            return statements

    def verify_statements(self, summary_statements: List[str], text: str) -> Dict[str, List[str]]:
        """
            Verify summary statements against source text.

            Implements the Verdict Reasoning phase from the paper's pipeline.
            Classifies each statement as:
            - TP (True Positive): Statement supported by the source text
            - FP (False Positive): Statement not supported by the source text

            Args:
                summary_statements (List[str]): List of statements extracted from the summary
                text (str): The original source text

            Returns:
                Dict[str, List[str]]: Classification of statements into TP and FP categories
            """

        prompt = """
        Compare the following statements from the summary with the statements from the original text. Do not add any ponctuation or words. Do not justify your answer.
        Label each summary statement as:
        - TP (True Positive): If the statement appears in the summary and is directly supported by a statement from the original text.
        - FP (False Positive): If the statement appears in the summary but is not directly supported by a statement from the original text.
        
        As you can see in the example bellow, first you have to concatenate the summary statements and the original text statements. 
        Then you have to label each statement as TP, FP or FN. Format as follow: VERDICT: TP, VERDICT: FP, VERDICT: FN
        
        Example: 
        Summary Statements:
        "Albert Einstein was born in Germany",
        "Albert Einstein was born in 1879"
        
        Original Text Statements:
        "Albert Einstein was born in Spain, in the city of Barcelona in 1879"
        
        
        Labels:
        
        Albert Einstein was born in 1879. VERDICT: TP
        Albert Einstein was born in Germany. VERDICT: FP
        
        END Example
        
        
        
        Summary Statements:
        {summary_statements}
        
        Original Text:
        {text}
        
        Labels:
        """.format(
            summary_statements="\n".join(summary_statements),
            text=text
        )

        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={"temperature": self.temperature}
        )

        # Update parsing to include FN
        labels = {"TP": [], "FP": [], "FN": []}
        for line in response['response'].split('\n'):
            line = line.strip()
            if "VERDICT: TP" in line or ' TP' in line:
                labels["TP"].append(line)
            elif "VERDICT: FP" in line or ' FP' in line:
                labels["FP"].append(line)
            elif "VERDICT: FN" in line or ' FN' in line:
                labels["FN"].append(line)

        return labels

    def calculate_metrics(self, labels: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Calculate evaluation metrics from labeled statements.

        Computes precision based on the ratio of true positives to all retrieved statements.
        According to the paper (formula 4), a comprehensive evaluation should include
        precision, recall, and F1 score.

        Args:
            labels (Dict[str, List[str]]): Classification of statements into categories

        Returns:
            Dict[str, float]: Dictionary of computed metrics
        """
        tp = len(labels["TP"])
        fp = len(labels["FP"])
        fn = len(labels["FN"]) if "FN" in labels else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp_count": tp,
            "fp_count": fp,
            "fn_count": fn
        }

    def evaluate_single(self, text: str, summary: str, extract_source_statements: bool = False) -> Dict:
        """
            Evaluate a single summary against its source text.

            Implements the full SEval-Ex pipeline for a single text-summary pair:
            1. Extract statements from summary
            2. Verify statements against source text
            3. Calculate evaluation metrics

            Args:
                text (str): The source document
                summary (str): The summary to be evaluated
                extract_source_statements (bool): Whether to explicitly extract statements from source
            When False, uses StSum_Text approach for direct comparison
            When True, extracts statements from both summary and source (Base approach)

            Returns:
                Dict: Evaluation results including extracted statements, verdict reasoning, and metrics
        """

        # Extract summary statements
        summary_statements = self.extract_statements(summary)

        # Extract source statements or use direct text comparison
        if extract_source_statements and self.method != "StSum_Text":
            source_statements = self.extract_statements(text)
            labels = self.verify_statements_with_source(summary_statements, source_statements)
        else:
            # Direct comparison (StSum_Text approach)
            labels = self.verify_statements(summary_statements, text)

        # Calculate metrics
        metrics = self.calculate_metrics(labels)

        return {
            "text": text,
            "summary": summary,
            "summary_statements": summary_statements,
            "source_statements": source_statements if extract_source_statements else [],
            "verdict_reasoning": labels,
            "metrics": metrics,
            "method": self.method,
            "chunked": len(text.split()) > self.max_text_length
        }

    def evaluate_dataset(self, dataset, text_key: str = "text", summary_key: str = "summary") -> List[Dict]:
        """
            Evaluate all examples in a dataset.

            Processes each text-summary pair in the dataset through the SEval-Ex pipeline.

            Args:
                dataset: The dataset containing text-summary pairs
                text_key (str): Key for accessing source text in dataset
                summary_key (str): Key for accessing summary in dataset

            Returns:
                List[Dict]: List of evaluation results for each text-summary pair
        """

        results = []
        for example in tqdm(dataset):
            for summary in example['machine_summaries']:
                result = self.evaluate_single(
                    text=example[text_key],
                    summary=summary
                )
                results.append(result)

            break
        return results


def main():
    parser = argparse.ArgumentParser(description="SEval Summary Evaluation Pipeline")
    parser.add_argument("--model", default="qwen2.5:72b", help="Model name to use")
    parser.add_argument("--method", default="StSum_Text", choices=["StSum_Text", "Base"],
                        help="Evaluation method")
    parser.add_argument("--chunk-size", type=int, default=3,
                        help="Number of sentences per chunk for long texts")
    parser.add_argument("--max-text-length", type=int, default=500,
                        help="Maximum word count before chunking is applied")
    parser.add_argument("--dataset", help="Dataset to evaluate (optional)")
    parser.add_argument("--text", help="Source text for evaluation (if no dataset)")
    parser.add_argument("--summary", help="Summary text for evaluation (if no dataset)")
    parser.add_argument("--output", default="seval_results.json", help="Output file path")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = SEvalPipeline(
        model_name=args.model,
        method=args.method,
        chunk_size=args.chunk_size,
        max_text_length=args.max_text_length
    )

    if args.dataset:
        # Dataset evaluation mode
        print(f"Loading dataset: {args.dataset}")
        dataset = load_dataset(args.dataset)
        print("Starting dataset evaluation...")
        results = pipeline.evaluate_dataset(dataset['test'])

        # Save results
        print(f"Saving results to {args.output}")
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        # Print summary
        print("\nDataset Evaluation Summary:")
        avg_precision = sum(r['metrics']['precision'] for r in results) / len(results)
        print(f"Average Precision: {avg_precision:.3f}")

    elif args.text and args.summary:
        # Single text evaluation mode
        print("Evaluating single text/summary pair...")
        result = pipeline.evaluate_single(args.text, args.summary)

        # Save result
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)

        # Print results
        print("\nEvaluation Results:")
        print(f"Precision: {result['metrics']['precision']:.3f}")
        print(f"True Positives: {result['metrics']['tp_count']}")
        print(f"False Positives: {result['metrics']['fp_count']}")
        print("\nExtracted Statements:")
        for stmt in result['summary_statements']:
            print(f"- {stmt}")
        print("\nVerdict Reasoning:")
        print("True Positives:")
        for tp in result['verdict_reasoning']['TP']:
            print(f"+ {tp}")
        print("False Positives:")
        for fp in result['verdict_reasoning']['FP']:
            print(f"- {fp}")

    else:
        parser.error("Either provide a dataset or both text and summary arguments")


if __name__ == "__main__":
    main()