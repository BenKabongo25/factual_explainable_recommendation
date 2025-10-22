# Statement-Topic-Sentiment Extraction (STS)

![](assets/statement_topic_sentiment.drawio.png)

Atomic explanatory statements from product reviews — with topic taxonomy, sentiment labels, embeddings and quantization.

> Convert raw reviews into **fine-grained atomic statements** labeled with a **domain-specific topic** and **sentiment**, then build **sentence embeddings** and **quantization** of paraphrases for downstream tasks (ABSA, ranking, scrutable recommendation, retrieval).

## News

## Domains & taxonomies

Each domain uses a **fixed topic list** and a unified JSON schema:

```json
[
  {"statement": "...", "topic": "<topic>", "sentiment": "positive|negative|neutral"}
]
```

* **Toys**: safety, age, durability, educational, engagement, materials, functionality, assembly, battery, price.&#x20;
* **Clothing**: fit, material, comfort, appearance, construction, price, care, functionality, shipping, service.&#x20;
* **Beauty**: efficacy, compatibility, ingredients, texture, longevity, color, price, packaging, scent, service.&#x20;
* **Sports**: performance, durability, comfort, fit, material, safety, portability, usability, price, service.&#x20;
* **Cellphones & accessories**: performance, battery, camera, display, specs, durability, software, usability, price, service.&#x20;

Each prompt file also specifies **atomicity rules**, **topic assignment rules**, and **few-shot examples** to ensure consistent extraction.

## Quick start (Toys example)

### 1) Extract atomic statements (LLM)

```bash
cd statement_topic_sentiment

PYTHONPATH=. python sts_extraction.py \
  --model /path/to/Meta-Llama-3-8B-Instruct \
  --prompt_text_file /path/to/prompt.txt \
  --dataset_path /path/to/reviews.json \
  --output_dir /path/to/topics/ \
  --batch_size 32 \
  --max_new_tokens 512
```

**Input format:** JSON/CSV with a review text field (e.g., `review`). The script loads rows, prompts the LLM with the domain prompt, and writes **per-review JSON arrays** of atomic statements following the schema above (topics & examples defined in `toys.txt`).&#x20;

### 2) Build statement embeddings

**Before embedding:** generate `statement_topic_sentiment_freq.csv` from your raw extraction outputs using the **post-processing notebooks** (see the `notebooks/` directory). These notebooks show how to:

* validate the schema (`statement`, `topic`, `sentiment`),
* deduplicate statements and compute a `freq` column.

```bash
PYTHONPATH=. python embed.py \
  --dataset_dir /path/to/topics \
  --model_name all-MiniLM-L6-v2 \
  --batch_size 32
```

This script then reads `statement_topic_sentiment_freq.csv`, encodes the **statement** column with Sentence-Transformers (L2-normalized), and saves `vectors/<model_name>/embeddings.pt`.


### 3) Quantization

```bash
PYTHONPATH=. python quantize.py \
    --embedding_path /path/to/embeddings.pt \
    --sts_csv /path/to/statement_topic_sentiment_freq.csv \
    --output_dir /path/to/rvq \
    --epochs 10 --batch_size 4096 \
    --codebook_size 32 --num_quantizers 3
```

---

## Apply Sentence-Topic-Sentiment to **your** dataset (and new domains)

1. **Write your own domain prompt** (`your_domain.txt`). Define the **accepted topics**, the **atomicity & formatting rules**, **sentiment guidelines**, and include **few-shot examples**. Keep the JSON schema exactly:

   ```json
   [
     {"statement": "...", "topic": "<topic>", "sentiment": "positive|negative|neutral"}
   ]
   ```

   **Minimal prompt skeleton**

   ```txt
   # <Your Domain> — Extraction Prompt (your_domain.txt)

   ACCEPTED TOPICS:
   - topic_1 — short definition
   - topic_2 — short definition
   - ...

   RULES:
   - Split reviews into atomic, present-tense factual statements (one claim per line).
   - Assign exactly one topic from ACCEPTED TOPICS to each statement.
   - Output a single valid JSON array (no prose, no trailing commas).
   - Use only "positive", "negative", or "neutral" for sentiment.

   SENTIMENT GUIDELINES:
   - positive: ...
   - negative: ...
   - neutral: ...

   OUTPUT FORMAT (MUST be valid JSON):
   [
     {"statement": "...", "topic": "<one_of_ACCEPTED_TOPICS>", "sentiment": "positive|negative|neutral"}
   ]

   EXAMPLES:
   Review: "..."
   Output: [
     {"statement": "...", "topic": "topic_1", "sentiment": "positive"},
     ...
   ]
   ```

2. **Map your schema** so your review text is available as one field (e.g., `review`).

3. **Run extraction** with `sts_extraction.py` using your prompt:

   ```bash
   PYTHONPATH=. python main.py \
     --model /path/to/instruct-model \
     --prompt_text_file /path/to/your_domain_prompt.txt \
     --dataset_path /path/to/reviews.json \
     --output_dir /path/to/topics/ \
     --batch_size 32 --max_new_tokens 512
   ```

   Verify that outputs are well-formed JSON arrays with keys `statement`, `topic`, `sentiment`.

4. **Aggregate** (optional) statement counts into `statement_topic_sentiment_freq.csv` (required columns: `statement,topic,sentiment`; optional `freq`).

5. **Run embeddings** and **quantization** to obtain vectors and quantization.

---

## Illustrative extraction examples

* **Example 1** (From `toys.txt` few-shot guidance.)

  Review: `The plastic edges are sharp and a small piece detached, and the wheels break during play.`
  Output:

  ```json
  [
    {"statement": "has sharp plastic edges", "topic": "safety", "sentiment": "negative"},
    {"statement": "has a small detachable part", "topic": "safety", "sentiment": "negative"},
    {"statement": "wheels break during play", "topic": "durability", "sentiment": "negative"}
  ]
  ```

