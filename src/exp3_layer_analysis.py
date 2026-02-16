"""
Supplementary analysis: Experiment 3 across different layers.
Also applies centering per-language to XLM-R contextualized embeddings.
"""

import json
import sys
import random
import numpy as np
import torch
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModel
from scipy import stats
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path(__file__).parent.parent
DATASETS = WORKSPACE / "datasets"
RESULTS = WORKSPACE / "results"
DEVICE = "cuda"


def load_mcl_wic_crosslingual(lang_pair):
    data_file = DATASETS / "mcl-wic" / "data" / "MCL-WiC" / "test" / "crosslingual" / f"test.{lang_pair}.data"
    gold_file = DATASETS / "mcl-wic" / "data" / f"test.{lang_pair}.gold"
    with open(data_file, "r") as f:
        data = json.load(f)
    with open(gold_file, "r") as f:
        gold = json.load(f)
    labels = {item["id"]: item["tag"] for item in gold}
    tags = [labels[item["id"]] for item in data]
    return data, tags


def parse_range(range_str):
    first_span = range_str.split(",")[0]
    parts = first_span.split("-")
    return (int(parts[0]), int(parts[1]))


def cosine_sim(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def extract_contextualized(model, tokenizer, sentences, char_ranges, layer, device, batch_size=32):
    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch_sents = sentences[i:i + batch_size]
        batch_ranges = char_ranges[i:i + batch_size]
        inputs = tokenizer(batch_sents, return_tensors="pt", padding=True,
                          truncation=True, max_length=256, return_offsets_mapping=True)
        offset_mapping = inputs.pop("offset_mapping")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad(), autocast('cuda'):
            outputs = model(**inputs)
        hidden_states = outputs.hidden_states
        layer_output = hidden_states[layer].float()
        for j in range(len(batch_sents)):
            char_start, char_end = batch_ranges[j]
            offsets = offset_mapping[j]
            token_mask = torch.zeros(offsets.size(0), dtype=torch.bool)
            for k in range(offsets.size(0)):
                tok_start, tok_end = offsets[k].tolist()
                if tok_end == 0:
                    continue
                if tok_start < char_end and tok_end > char_start:
                    token_mask[k] = True
            if token_mask.sum() == 0:
                attention_mask = inputs["attention_mask"][j]
                token_mask = attention_mask.bool()
                token_mask[0] = False
                last_pos = attention_mask.sum().item() - 1
                if last_pos > 0:
                    token_mask[last_pos] = False
            target_embeddings = layer_output[j][token_mask]
            mean_emb = target_embeddings.mean(dim=0).cpu().numpy()
            all_embeddings.append(mean_emb)
    return np.stack(all_embeddings, axis=0)


def main():
    results = {}
    models = {
        "mbert": "bert-base-multilingual-cased",
        "xlmr": "xlm-roberta-base",
    }
    mcl_pairs = ["en-fr", "en-zh", "en-ru"]
    layers_to_test = [1, 4, 7, 10, -1]

    for model_key, model_name in models.items():
        logger.info(f"\n=== Model: {model_key} ===")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        model.to(DEVICE).eval()

        results[model_key] = {}

        for lang_pair in mcl_pairs:
            logger.info(f"  Language pair: {lang_pair}")
            data, tags = load_mcl_wic_crosslingual(lang_pair)

            sentences1, char_ranges1 = [], []
            sentences2, char_ranges2 = [], []
            for item in data:
                sentences1.append(item["sentence1"])
                sentences2.append(item["sentence2"])
                if "start1" in item:
                    char_ranges1.append((int(item["start1"]), int(item["end1"])))
                elif "ranges1" in item:
                    char_ranges1.append(parse_range(item["ranges1"]))
                else:
                    char_ranges1.append((0, len(item["sentence1"])))
                if "start2" in item:
                    char_ranges2.append((int(item["start2"]), int(item["end2"])))
                elif "ranges2" in item:
                    char_ranges2.append(parse_range(item["ranges2"]))
                else:
                    char_ranges2.append((0, len(item["sentence2"])))

            layer_results = {}
            for layer in layers_to_test:
                embs1 = extract_contextualized(model, tokenizer, sentences1, char_ranges1, layer, DEVICE)
                embs2 = extract_contextualized(model, tokenizer, sentences2, char_ranges2, layer, DEVICE)

                # Center embeddings per-group (embs1 = English, embs2 = foreign)
                embs1_c = embs1 - embs1.mean(axis=0, keepdims=True)
                embs2_c = embs2 - embs2.mean(axis=0, keepdims=True)

                same_sims, diff_sims = [], []
                for i in range(len(data)):
                    sim = cosine_sim(embs1_c[i], embs2_c[i])
                    if tags[i] == "T":
                        same_sims.append(sim)
                    else:
                        diff_sims.append(sim)

                same_sims = np.array(same_sims)
                diff_sims = np.array(diff_sims)
                stat, pval = stats.mannwhitneyu(same_sims, diff_sims, alternative='greater')
                d = (same_sims.mean() - diff_sims.mean()) / np.sqrt(
                    (same_sims.std()**2 + diff_sims.std()**2) / 2)

                layer_results[str(layer)] = {
                    "same_mean": float(same_sims.mean()),
                    "diff_mean": float(diff_sims.mean()),
                    "cohens_d": float(d),
                    "p_value": float(pval),
                }
                logger.info(f"    Layer {layer}: same={same_sims.mean():.4f}, diff={diff_sims.mean():.4f}, d={d:.3f}")

            results[model_key][lang_pair] = layer_results

        del model, tokenizer
        torch.cuda.empty_cache()

    with open(RESULTS / "exp3_layer_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved exp3_layer_analysis.json")


if __name__ == "__main__":
    main()
