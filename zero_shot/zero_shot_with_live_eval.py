import json
import time
import re
from groq import Groq
from sklearn.metrics import f1_score, precision_recall_fscore_support

# ================= CONFIG =================
INPUT_FILE = "../data-preprocessing/basil_consolidated_all.json"
OUTPUT_FILE = "zero_shot_article_level_outputs.json"
MODEL = "llama-3.3-70b-versatile"
LIMIT = 25
DELAY = 2

client = Groq()

# ================= METRICS =================

class MetricsTracker:
    def __init__(self):
        self.y_true = []
        self.y_pred = []
        self.articles_processed = 0
        self.parse_failures = 0

    def add_prediction(self, gt, pred):
        self.y_true.append(gt)
        self.y_pred.append(pred)
        self.articles_processed += 1

    def get_metrics(self):
        if not self.y_true:
            return None

        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_true,
            self.y_pred,
            average="binary",
            zero_division=0
        )

        f1_macro = f1_score(
            self.y_true,
            self.y_pred,
            average="macro",
            zero_division=0
        )

        return {
            "articles": self.articles_processed,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f1_macro": f1_macro
        }

    def print_progress(self):
        m = self.get_metrics()
        if m:
            print(
                f"  Articles: {m['articles']} | "
                f"F1: {m['f1']:.3f} | "
                f"P: {m['precision']:.3f} | "
                f"R: {m['recall']:.3f}"
            )

# ================= HELPERS =================

def extract_json_from_response(text):
    if not text:
        return None

    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def get_gt_article_label(article):
    """
    Ground truth:
    Article is biased if ANY sentence is biased.
    """
    for s in article.get("sentences", []):
        if s.get("bias") != "No":
            return 1
    return 0


def get_pred_article_label(parsed_output):
    return 1 if parsed_output.get("article_bias") == "biased" else 0


# ================= PROMPT =================

def format_article_prompt(article):
    meta = article.get("article_metadata", {})
    sentences = [s["text"] for s in article.get("sentences", [])][:40]
    article_text = "\n".join(sentences)

    instruction = (
        "You are an expert in political linguistics and media bias analysis.\n\n"
        "Analyze the following news article AS A WHOLE.\n\n"
        "Decide:\n"
        "1. article_bias:\n"
        "   - 'biased' if the article uses subjective framing, emotionally loaded language,\n"
        "     selective emphasis, or implicit judgment\n"
        "   - 'neutral' if the article is objective and factual\n\n"
        "2. orientation (if biased): 'liberal', 'conservative', or 'neutral'\n"
        "3. rationale: 1–2 sentences justifying your decision\n\n"
        "Return VALID JSON ONLY in this exact format:\n"
        '{ "article_bias": "neutral", "orientation": "neutral", "rationale": "..." }'
    )

    article_info = (
        f"\nTitle: {meta.get('title', 'unknown')}\n"
        f"Source: {meta.get('source', 'unknown')}\n\n"
        f"Article Text:\n{article_text}"
    )

    return f"{instruction}\n{article_info}"


def analyze_article(article):
    prompt = format_article_prompt(article)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=3000,
        )

        if not response.choices:
            return None, None

        output_text = response.choices[0].message.content.strip()
        parsed = extract_json_from_response(output_text)

        return output_text, parsed

    except Exception as e:
        print(f"  [ERROR] {e}")
        return None, None


# ================= MAIN =================

def run_zero_shot_with_eval():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} articles")
    print(f"Processing up to {LIMIT} articles\n")

    tracker = MetricsTracker()
    results = []

    for i, article in enumerate(data[:LIMIT]):
        title = article.get("article_metadata", {}).get("title", "Untitled")
        print(f"\n[{i+1}/{LIMIT}] {title[:70]}")

        raw, parsed = analyze_article(article)

        if not parsed:
            tracker.parse_failures += 1
            print("  ✗ Parse failure")
            continue

        gt = get_gt_article_label(article)
        pred = get_pred_article_label(parsed)

        tracker.add_prediction(gt, pred)
        tracker.print_progress()

        results.append({
            "uuid": article.get("uuid"),
            "triplet_uuid": article.get("triplet_uuid"),
            "title": title,
            "ground_truth_article_bias": "biased" if gt else "neutral",
            "model_output": parsed,
            "raw_output": raw
        })

        if i < LIMIT - 1:
            time.sleep(DELAY)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    metrics = tracker.get_metrics()
    if metrics:
        print(f"Articles Processed: {metrics['articles']}")
        print(f"Parse Failures: {tracker.parse_failures}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall:    {metrics['recall']:.3f}")
        print(f"F1 Score:  {metrics['f1']:.3f}")
        print(f"F1 Macro:  {metrics['f1_macro']:.3f}")

    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_zero_shot_with_eval()
