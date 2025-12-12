import json
import re
from tqdm import tqdm
from textblob import TextBlob
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "Babelscape/rebel-large"   # Fast + reliable
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# ARTICLE BIAS NORMALIZATION
# ============================================================
BIAS_NORMALIZATION_MAP = {
    "left": "left",
    "liberal": "left",
    "center": "center",
    "right": "right",
    "conservative": "right"
}

def normalize_article_bias(bias_label):
    """
    Normalize article bias → left / center / right
    """
    if not bias_label:
        return "center"

    bias_lower = bias_label.lower().strip()
    return BIAS_NORMALIZATION_MAP.get(bias_lower, "center")


# ============================================================
# LOAD REBEL MODEL
# ============================================================
print("🔧 Loading REBEL model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


# ============================================================
# BATCH REBEL EXTRACTION
# ============================================================
def rebel_extract_batch(sentences, max_length=128):
    encoded = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_length=max_length,
            num_beams=3,
            early_stopping=True
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=False)


# ============================================================
# PARSE REBEL "<triplet>" TEXT INTO (actor, predicate, object)
# ============================================================
def parse_rebel_text(text):
    triples = []
    pattern = r"<triplet>\s*(.*?)\s*<subj>\s*(.*?)\s*<obj>\s*(.*?)(?=<triplet>|$)"

    for match in re.findall(pattern, text, flags=re.DOTALL):
        subj = match[0].strip()
        pred = match[1].strip()
        obj = match[2].strip()

        # Clean noise
        subj = subj.replace("<pad>", "").strip()
        pred = pred.split()[0].strip()   # predicate head only
        obj = obj.replace("<pad>", "").strip()

        if subj and pred and obj:
            triples.append((subj, pred, obj))

    return triples


# ============================================================
# BIAS LOOKUPS
# ============================================================
def build_bias_lookup(annotations):
    lookup = {}
    for ann in annotations:
        target = ann.get("target", "")
        lookup[target] = {
            "text": ann.get("txt", ""),
            "type": ann.get("bias", ""),
            "polarity": ann.get("polarity", "")
        }
    return lookup


def attach_bias_info(event, bias_info):
    trig = event["trigger"].lower()
    actor = event["actor"].lower()
    obj = event["object"].lower()

    for target, info in bias_info.items():
        word = info["text"].lower()
        if word and (word in trig or word in actor or word in obj):
            return {
                "has_bias": True,
                "bias_type": info["type"],
                "polarity": info["polarity"],
                "target": target,
                "biased_word": info["text"]
            }

    return {
        "has_bias": False,
        "bias_type": None,
        "polarity": None,
        "target": None,
        "biased_word": None
    }


# ============================================================
# MAIN PIPELINE
# ============================================================
def refine_step2(input_path, output_path, batch_size=12):
    data = json.load(open(input_path))
    print(f"⚙️ Step 2: Extracting events for {len(data)} sentences...")

    texts = [s["text"] for s in data]
    all_triples = []

    # ---- Batch process with REBEL ----
    for i in tqdm(range(0, len(texts), batch_size), desc="REBEL batches"):
        batch = texts[i:i+batch_size]
        outputs = rebel_extract_batch(batch)

        for out in outputs:
            triples = parse_rebel_text(out)
            all_triples.append(triples)

    # ---- Attach events + metadata ----
    refined_data = []
    for idx, sent in enumerate(data):
        triples = all_triples[idx]
        bias_info = build_bias_lookup(sent.get("annotations", []))
        sentiment = round(TextBlob(sent["text"]).sentiment.polarity, 3)

        # Normalize article bias
        article_bias = normalize_article_bias(sent.get("article_bias", ""))

        events = []
        for (actor, pred, obj) in triples:
            bias = attach_bias_info({
                "trigger": pred,
                "actor": actor,
                "object": obj
            }, bias_info)

            events.append({
                "trigger": pred,
                "actor": actor,
                "object": obj,
                "sentiment": sentiment,
                **bias
            })

        sent_out = {
            **sent,
            "sentence_idx": idx,
            "article_bias_normalized": article_bias,
            "events": events
        }

        refined_data.append(sent_out)

    # ---- Save output ----
    json.dump(refined_data, open(output_path, "w"), indent=2, ensure_ascii=False)
    print("✅ Step 2 complete — saved:", output_path)


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    refine_step2(
        # input_path="../news-discourse/toy_discourse_example.json",
        # output_path="toy_discourse_with_events.json"
        input_path="../news-discourse/discourse_data.json",
        output_path="discourse_with_events.json"
    )