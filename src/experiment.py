"""
Cross-Lingual Embedding Similarity & Polysemy Experiments
=========================================================
Tests whether words with similar meanings in different languages have similar
embeddings in multilingual transformer models, and how polysemy affects this.

Experiments:
1. Type-level similarity of translation pairs vs. random pairs
2. Monosemous vs. polysemous translation pair similarity
3. Contextualized sense-level similarity (MCL-WiC)
4. Cross-lingual word similarity (SemEval-2017 Task 2)
"""

import os
import sys
import json
import random
import logging
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModel
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist
from nltk.corpus import wordnet as wn
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────

SEED = 42
WORKSPACE = Path(__file__).parent.parent
DATASETS = WORKSPACE / "datasets"
RESULTS = WORKSPACE / "results"
RESULTS.mkdir(exist_ok=True)
(RESULTS / "plots").mkdir(exist_ok=True)

MODELS = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
}

LANG_PAIRS = ["en-fr", "en-de", "en-es", "en-ru", "en-zh"]
MAX_TRANSLATION_PAIRS = 5000  # per language pair for efficiency
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_muse_dict(lang_pair: str, max_pairs: int = MAX_TRANSLATION_PAIRS) -> list[tuple[str, str]]:
    """Load MUSE bilingual dictionary as list of (en_word, foreign_word) pairs."""
    filepath = DATASETS / "muse" / f"{lang_pair}.txt"
    pairs = []
    seen = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
            en, foreign = parts
            # Keep only single-word, alphabetic entries (no phrases or punctuation)
            if " " in en or " " in foreign:
                continue
            if not en.isalpha() or not foreign.replace("'", "").replace("-", "").replace("'", "").isalpha():
                continue
            key = (en.lower(), foreign.lower())
            if key not in seen:
                seen.add(key)
                pairs.append((en, foreign))
    random.shuffle(pairs)
    return pairs[:max_pairs]


def get_wordnet_sense_count(word: str) -> int:
    """Get number of WordNet synsets for an English word."""
    synsets = wn.synsets(word.lower(), lang='eng')
    return len(synsets)


def classify_polysemy(pairs: list[tuple[str, str]]) -> dict:
    """Classify translation pairs by English word polysemy level."""
    result = {"monosemous": [], "polysemous": [], "highly_polysemous": []}
    sense_counts = []
    for en, foreign in pairs:
        n_senses = get_wordnet_sense_count(en)
        sense_counts.append(n_senses)
        if n_senses == 1:
            result["monosemous"].append((en, foreign, n_senses))
        elif 2 <= n_senses <= 4:
            result["polysemous"].append((en, foreign, n_senses))
        elif n_senses >= 5:
            result["highly_polysemous"].append((en, foreign, n_senses))
    # Words with 0 senses (not in WordNet) are excluded
    logger.info(f"Polysemy classification: monosemous={len(result['monosemous'])}, "
                f"polysemous(2-4)={len(result['polysemous'])}, "
                f"highly_polysemous(5+)={len(result['highly_polysemous'])}")
    return result


def load_semeval_crosslingual(lang_pair: str) -> tuple[list[tuple[str, str]], list[float]]:
    """Load SemEval-2017 Task 2 cross-lingual word pairs and gold scores."""
    data_dir = DATASETS / "semeval2017-task2" / "SemEval17-Task2" / "test" / "subtask2-crosslingual"
    data_file = data_dir / "data" / f"{lang_pair}.test.data.txt"
    gold_file = data_dir / "keys" / f"{lang_pair}.test.gold.txt"

    pairs = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0].strip(), parts[1].strip()))

    scores = []
    with open(gold_file, "r", encoding="utf-8") as f:
        for line in f:
            scores.append(float(line.strip()))

    assert len(pairs) == len(scores), f"Mismatch: {len(pairs)} pairs vs {len(scores)} scores"
    return pairs, scores


def load_mcl_wic_crosslingual(lang_pair: str) -> tuple[list[dict], list[str]]:
    """Load MCL-WiC cross-lingual test data and gold labels."""
    data_file = DATASETS / "mcl-wic" / "data" / "MCL-WiC" / "test" / "crosslingual" / f"test.{lang_pair}.data"
    gold_file = DATASETS / "mcl-wic" / "data" / f"test.{lang_pair}.gold"

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(gold_file, "r", encoding="utf-8") as f:
        gold = json.load(f)

    labels = {item["id"]: item["tag"] for item in gold}
    tags = [labels[item["id"]] for item in data]
    return data, tags


# ─── Embedding Extraction ────────────────────────────────────────────────────

class EmbeddingExtractor:
    """Extract embeddings from multilingual transformer models."""

    def __init__(self, model_name: str, device: str = DEVICE):
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.model_name = model_name

    def get_word_embeddings(self, words: list[str], layer: int = -1, batch_size: int = BATCH_SIZE) -> np.ndarray:
        """
        Get type-level embeddings for isolated words.
        Uses mean-pooling of subword token embeddings (excluding special tokens).
        """
        all_embeddings = []
        for i in range(0, len(words), batch_size):
            batch = words[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=32)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad(), autocast('cuda', enabled=self.device == 'cuda'):
                outputs = self.model(**inputs)

            hidden_states = outputs.hidden_states  # tuple of (n_layers+1, batch, seq, dim)
            layer_output = hidden_states[layer]  # (batch, seq, dim)

            # Mean-pool over non-special, non-padding tokens
            attention_mask = inputs["attention_mask"]
            # Mask out [CLS] and [SEP] / <s> and </s>
            token_mask = attention_mask.clone()
            token_mask[:, 0] = 0  # mask CLS/<s>
            # Find last non-padding position per sequence and mask it (SEP/</s>)
            for j in range(token_mask.size(0)):
                last_pos = attention_mask[j].sum().item() - 1
                if last_pos > 0:
                    token_mask[j, last_pos] = 0

            # Expand mask for broadcasting
            mask_expanded = token_mask.unsqueeze(-1).float()
            sum_embeddings = (layer_output.float() * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            mean_embeddings = sum_embeddings / count

            all_embeddings.append(mean_embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def get_contextualized_embeddings(self, sentences: list[str], char_ranges: list[tuple[int, int]],
                                       layer: int = -1, batch_size: int = 32) -> np.ndarray:
        """
        Get contextualized embeddings for target words within sentences.
        char_ranges specifies the character span of the target word in each sentence.
        Returns mean-pooled subword embeddings for the target word tokens.
        """
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_sents = sentences[i:i + batch_size]
            batch_ranges = char_ranges[i:i + batch_size]

            inputs = self.tokenizer(batch_sents, return_tensors="pt", padding=True,
                                    truncation=True, max_length=256, return_offsets_mapping=True)

            offset_mapping = inputs.pop("offset_mapping")  # (batch, seq, 2)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad(), autocast('cuda', enabled=self.device == 'cuda'):
                outputs = self.model(**inputs)

            hidden_states = outputs.hidden_states
            layer_output = hidden_states[layer].float()  # (batch, seq, dim)

            for j in range(len(batch_sents)):
                char_start, char_end = batch_ranges[j]
                offsets = offset_mapping[j]  # (seq, 2)

                # Find tokens that overlap with target word character span
                token_mask = torch.zeros(offsets.size(0), dtype=torch.bool)
                for k in range(offsets.size(0)):
                    tok_start, tok_end = offsets[k].tolist()
                    if tok_end == 0:  # special token
                        continue
                    if tok_start < char_end and tok_end > char_start:
                        token_mask[k] = True

                if token_mask.sum() == 0:
                    # Fallback: use mean of all non-special tokens
                    attention_mask = inputs["attention_mask"][j]
                    token_mask = attention_mask.bool()
                    token_mask[0] = False
                    last_pos = attention_mask.sum().item() - 1
                    if last_pos > 0:
                        token_mask[last_pos] = False

                target_embeddings = layer_output[j][token_mask]  # (n_tokens, dim)
                mean_emb = target_embeddings.mean(dim=0).cpu().numpy()
                all_embeddings.append(mean_emb)

        return np.stack(all_embeddings, axis=0)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def center_embeddings(embs: np.ndarray) -> np.ndarray:
    """Center embeddings by subtracting the mean (Libovicky et al. 2020)."""
    return embs - embs.mean(axis=0, keepdims=True)


# ─── Experiment 1: Type-level similarity of translation pairs ────────────────

def experiment_1(extractor: EmbeddingExtractor, model_key: str):
    """Measure cosine similarity of translation pair embeddings vs random pairs."""
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT 1: Type-level translation pair similarity ({model_key})")
    logger.info("=" * 60)

    results = {}
    layers_to_test = [0, 3, 6, 9, -1]  # test multiple layers

    for lang_pair in LANG_PAIRS:
        logger.info(f"\n--- Language pair: {lang_pair} ---")
        pairs = load_muse_dict(lang_pair)
        if len(pairs) < 100:
            logger.warning(f"Too few pairs for {lang_pair}, skipping")
            continue

        en_words = [p[0] for p in pairs]
        foreign_words = [p[1] for p in pairs]
        n = len(pairs)
        logger.info(f"Loaded {n} translation pairs")

        layer_results = {}
        for layer in layers_to_test:
            # Extract embeddings
            en_embs = extractor.get_word_embeddings(en_words, layer=layer)
            foreign_embs = extractor.get_word_embeddings(foreign_words, layer=layer)

            # Center embeddings per language
            en_centered = center_embeddings(en_embs)
            foreign_centered = center_embeddings(foreign_embs)

            # Compute cosine similarity for translation pairs
            trans_sims = []
            for i in range(n):
                sim = cosine_sim(en_centered[i], foreign_centered[i])
                trans_sims.append(sim)

            # Compute cosine similarity for random pairs (shuffle foreign side)
            random_indices = list(range(n))
            random.shuffle(random_indices)
            random_sims = []
            for i in range(n):
                sim = cosine_sim(en_centered[i], foreign_centered[random_indices[i]])
                random_sims.append(sim)

            trans_sims = np.array(trans_sims)
            random_sims = np.array(random_sims)

            # Also compute raw (uncentered) similarities
            raw_trans_sims = np.array([cosine_sim(en_embs[i], foreign_embs[i]) for i in range(n)])
            raw_random_sims = np.array([cosine_sim(en_embs[i], foreign_embs[random_indices[i]]) for i in range(n)])

            # Statistical test
            stat, pval = stats.mannwhitneyu(trans_sims, random_sims, alternative='greater')
            effect_size = (trans_sims.mean() - random_sims.mean()) / np.sqrt(
                (trans_sims.std() ** 2 + random_sims.std() ** 2) / 2)

            layer_results[layer] = {
                "translation_sim_mean": float(trans_sims.mean()),
                "translation_sim_std": float(trans_sims.std()),
                "random_sim_mean": float(random_sims.mean()),
                "random_sim_std": float(random_sims.std()),
                "raw_translation_sim_mean": float(raw_trans_sims.mean()),
                "raw_random_sim_mean": float(raw_random_sims.mean()),
                "mann_whitney_stat": float(stat),
                "p_value": float(pval),
                "cohens_d": float(effect_size),
                "n_pairs": n,
            }

            logger.info(f"  Layer {layer}: trans_sim={trans_sims.mean():.4f}±{trans_sims.std():.4f}, "
                        f"random_sim={random_sims.mean():.4f}±{random_sims.std():.4f}, "
                        f"Cohen's d={effect_size:.3f}, p={pval:.2e}")

        results[lang_pair] = layer_results

    return results


# ─── Experiment 2: Monosemous vs. Polysemous ─────────────────────────────────

def experiment_2(extractor: EmbeddingExtractor, model_key: str):
    """Compare embedding similarity between monosemous and polysemous translation pairs."""
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT 2: Monosemous vs. polysemous similarity ({model_key})")
    logger.info("=" * 60)

    results = {}
    best_layer = -1  # Use last layer; can be refined based on Exp 1

    for lang_pair in LANG_PAIRS:
        logger.info(f"\n--- Language pair: {lang_pair} ---")
        pairs = load_muse_dict(lang_pair)
        classified = classify_polysemy(pairs)

        if len(classified["monosemous"]) < 30 or len(classified["polysemous"]) < 30:
            logger.warning(f"Too few classified pairs for {lang_pair}, skipping")
            continue

        # Build combined word lists for batch embedding
        all_en = []
        all_foreign = []
        categories = []
        sense_counts = []

        for cat in ["monosemous", "polysemous", "highly_polysemous"]:
            for en, foreign, n_senses in classified[cat]:
                all_en.append(en)
                all_foreign.append(foreign)
                categories.append(cat)
                sense_counts.append(n_senses)

        # Extract embeddings
        en_embs = extractor.get_word_embeddings(all_en, layer=best_layer)
        foreign_embs = extractor.get_word_embeddings(all_foreign, layer=best_layer)

        # Center
        en_centered = center_embeddings(en_embs)
        foreign_centered = center_embeddings(foreign_embs)

        # Compute similarities by category
        cat_sims = defaultdict(list)
        cat_sense_sims = defaultdict(list)  # for fine-grained sense count analysis
        for i in range(len(all_en)):
            sim = cosine_sim(en_centered[i], foreign_centered[i])
            cat_sims[categories[i]].append(sim)
            cat_sense_sims[sense_counts[i]].append(sim)

        # Statistical tests
        mono_sims = np.array(cat_sims["monosemous"])
        poly_sims = np.array(cat_sims["polysemous"])
        highly_poly_sims = np.array(cat_sims["highly_polysemous"]) if cat_sims["highly_polysemous"] else np.array([])

        # Mann-Whitney U test: monosemous vs polysemous
        stat, pval = stats.mannwhitneyu(mono_sims, poly_sims, alternative='greater')
        effect_d = (mono_sims.mean() - poly_sims.mean()) / np.sqrt(
            (mono_sims.std() ** 2 + poly_sims.std() ** 2) / 2)

        # Correlation between sense count and similarity
        all_sims = [cosine_sim(en_centered[i], foreign_centered[i]) for i in range(len(all_en))]
        spearman_r, spearman_p = stats.spearmanr(sense_counts, all_sims)

        result_entry = {
            "monosemous": {
                "n": len(mono_sims),
                "sim_mean": float(mono_sims.mean()),
                "sim_std": float(mono_sims.std()),
                "sim_median": float(np.median(mono_sims)),
            },
            "polysemous_2_4": {
                "n": len(poly_sims),
                "sim_mean": float(poly_sims.mean()),
                "sim_std": float(poly_sims.std()),
                "sim_median": float(np.median(poly_sims)),
            },
            "highly_polysemous_5plus": {
                "n": len(highly_poly_sims),
                "sim_mean": float(highly_poly_sims.mean()) if len(highly_poly_sims) > 0 else None,
                "sim_std": float(highly_poly_sims.std()) if len(highly_poly_sims) > 0 else None,
            },
            "mono_vs_poly_mannwhitney_stat": float(stat),
            "mono_vs_poly_pvalue": float(pval),
            "mono_vs_poly_cohens_d": float(effect_d),
            "sense_count_vs_sim_spearman_r": float(spearman_r),
            "sense_count_vs_sim_spearman_p": float(spearman_p),
            # Fine-grained by sense count
            "by_sense_count": {
                str(k): {"n": len(v), "mean_sim": float(np.mean(v)), "std_sim": float(np.std(v))}
                for k, v in sorted(cat_sense_sims.items()) if len(v) >= 5
            },
        }

        logger.info(f"  Monosemous (n={len(mono_sims)}): sim={mono_sims.mean():.4f}±{mono_sims.std():.4f}")
        logger.info(f"  Polysemous 2-4 (n={len(poly_sims)}): sim={poly_sims.mean():.4f}±{poly_sims.std():.4f}")
        if len(highly_poly_sims) > 0:
            logger.info(f"  Highly poly 5+ (n={len(highly_poly_sims)}): sim={highly_poly_sims.mean():.4f}±{highly_poly_sims.std():.4f}")
        logger.info(f"  Mann-Whitney U p={pval:.2e}, Cohen's d={effect_d:.3f}")
        logger.info(f"  Sense count vs sim: Spearman r={spearman_r:.3f}, p={spearman_p:.2e}")

        results[lang_pair] = result_entry

    return results


# ─── Experiment 3: Contextualized sense-level similarity (MCL-WiC) ───────────

def experiment_3(extractor: EmbeddingExtractor, model_key: str):
    """Test whether same-sense cross-lingual pairs have higher embedding similarity."""
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT 3: Contextualized sense-level similarity ({model_key})")
    logger.info("=" * 60)

    results = {}
    mcl_pairs = ["en-fr", "en-zh", "en-ru"]  # Available cross-lingual pairs with gold labels

    for lang_pair in mcl_pairs:
        logger.info(f"\n--- Language pair: {lang_pair} ---")
        try:
            data, tags = load_mcl_wic_crosslingual(lang_pair)
        except FileNotFoundError as e:
            logger.warning(f"Missing data for {lang_pair}: {e}")
            continue

        n = len(data)
        logger.info(f"Loaded {n} examples ({tags.count('T')} same-sense, {tags.count('F')} diff-sense)")

        # Extract contextualized embeddings for both sentences
        sentences1 = []
        char_ranges1 = []
        sentences2 = []
        char_ranges2 = []

        for item in data:
            sentences1.append(item["sentence1"])
            sentences2.append(item["sentence2"])

            # Parse char ranges (can be "start-end" or "start-end,start2-end2" for discontinuous)
            def parse_range(range_str):
                """Parse range string, taking first span if discontinuous."""
                first_span = range_str.split(",")[0]
                parts = first_span.split("-")
                return (int(parts[0]), int(parts[1]))

            if "start1" in item and "end1" in item:
                char_ranges1.append((int(item["start1"]), int(item["end1"])))
            elif "ranges1" in item:
                char_ranges1.append(parse_range(item["ranges1"]))
            else:
                char_ranges1.append((0, len(item["sentence1"])))

            if "start2" in item and "end2" in item:
                char_ranges2.append((int(item["start2"]), int(item["end2"])))
            elif "ranges2" in item:
                char_ranges2.append(parse_range(item["ranges2"]))
            else:
                char_ranges2.append((0, len(item["sentence2"])))

        embs1 = extractor.get_contextualized_embeddings(sentences1, char_ranges1)
        embs2 = extractor.get_contextualized_embeddings(sentences2, char_ranges2)

        # Compute similarities
        same_sense_sims = []
        diff_sense_sims = []
        all_sims = []

        for i in range(n):
            sim = cosine_sim(embs1[i], embs2[i])
            all_sims.append(sim)
            if tags[i] == "T":
                same_sense_sims.append(sim)
            else:
                diff_sense_sims.append(sim)

        same_sense_sims = np.array(same_sense_sims)
        diff_sense_sims = np.array(diff_sense_sims)

        # Statistical test
        stat, pval = stats.mannwhitneyu(same_sense_sims, diff_sense_sims, alternative='greater')
        effect_d = (same_sense_sims.mean() - diff_sense_sims.mean()) / np.sqrt(
            (same_sense_sims.std() ** 2 + diff_sense_sims.std() ** 2) / 2)

        # Classification accuracy using similarity threshold
        all_sims_arr = np.array(all_sims)
        all_tags = np.array([1 if t == "T" else 0 for t in tags])
        best_acc = 0
        best_thresh = 0
        for thresh in np.arange(0.0, 1.0, 0.01):
            preds = (all_sims_arr >= thresh).astype(int)
            acc = (preds == all_tags).mean()
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        results[lang_pair] = {
            "n_total": n,
            "n_same_sense": len(same_sense_sims),
            "n_diff_sense": len(diff_sense_sims),
            "same_sense_sim_mean": float(same_sense_sims.mean()),
            "same_sense_sim_std": float(same_sense_sims.std()),
            "diff_sense_sim_mean": float(diff_sense_sims.mean()),
            "diff_sense_sim_std": float(diff_sense_sims.std()),
            "mann_whitney_stat": float(stat),
            "p_value": float(pval),
            "cohens_d": float(effect_d),
            "best_threshold_accuracy": float(best_acc),
            "best_threshold": float(best_thresh),
        }

        logger.info(f"  Same-sense (n={len(same_sense_sims)}): sim={same_sense_sims.mean():.4f}±{same_sense_sims.std():.4f}")
        logger.info(f"  Diff-sense (n={len(diff_sense_sims)}): sim={diff_sense_sims.mean():.4f}±{diff_sense_sims.std():.4f}")
        logger.info(f"  Mann-Whitney p={pval:.2e}, Cohen's d={effect_d:.3f}")
        logger.info(f"  Best threshold accuracy: {best_acc:.4f} (threshold={best_thresh:.2f})")

    return results


# ─── Experiment 4: Cross-lingual word similarity (SemEval-2017) ──────────────

def experiment_4(extractor: EmbeddingExtractor, model_key: str):
    """Evaluate cross-lingual word similarity on SemEval-2017 Task 2."""
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT 4: Cross-lingual word similarity SemEval-2017 ({model_key})")
    logger.info("=" * 60)

    results = {}
    semeval_pairs = ["en-de", "en-es", "en-it", "en-fa"]

    for lang_pair in semeval_pairs:
        logger.info(f"\n--- Language pair: {lang_pair} ---")
        try:
            word_pairs, gold_scores = load_semeval_crosslingual(lang_pair)
        except FileNotFoundError as e:
            logger.warning(f"Missing data for {lang_pair}: {e}")
            continue

        n = len(word_pairs)
        logger.info(f"Loaded {n} word pairs")

        # Extract embeddings
        words1 = [p[0] for p in word_pairs]
        words2 = [p[1] for p in word_pairs]

        embs1 = extractor.get_word_embeddings(words1, layer=-1)
        embs2 = extractor.get_word_embeddings(words2, layer=-1)

        # Center per language
        embs1_c = center_embeddings(embs1)
        embs2_c = center_embeddings(embs2)

        # Compute cosine similarities
        pred_sims_centered = [cosine_sim(embs1_c[i], embs2_c[i]) for i in range(n)]
        pred_sims_raw = [cosine_sim(embs1[i], embs2[i]) for i in range(n)]

        # Correlate with gold scores
        spearman_centered, sp_p_centered = stats.spearmanr(pred_sims_centered, gold_scores)
        spearman_raw, sp_p_raw = stats.spearmanr(pred_sims_raw, gold_scores)
        pearson_centered, pe_p_centered = stats.pearsonr(pred_sims_centered, gold_scores)

        results[lang_pair] = {
            "n_pairs": n,
            "spearman_centered": float(spearman_centered),
            "spearman_centered_p": float(sp_p_centered),
            "spearman_raw": float(spearman_raw),
            "spearman_raw_p": float(sp_p_raw),
            "pearson_centered": float(pearson_centered),
            "pearson_centered_p": float(pe_p_centered),
        }

        logger.info(f"  Spearman (centered): {spearman_centered:.4f} (p={sp_p_centered:.2e})")
        logger.info(f"  Spearman (raw): {spearman_raw:.4f} (p={sp_p_raw:.2e})")
        logger.info(f"  Pearson (centered): {pearson_centered:.4f}")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    set_seed()
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, "
                f"Device: {DEVICE}, GPUs: {torch.cuda.device_count()}")

    all_results = {}

    for model_key, model_name in MODELS.items():
        logger.info(f"\n{'#' * 70}")
        logger.info(f"# MODEL: {model_key} ({model_name})")
        logger.info(f"{'#' * 70}\n")

        extractor = EmbeddingExtractor(model_name, device=DEVICE)

        all_results[model_key] = {
            "exp1_translation_similarity": experiment_1(extractor, model_key),
            "exp2_monosemous_vs_polysemous": experiment_2(extractor, model_key),
            "exp3_contextualized_sense_similarity": experiment_3(extractor, model_key),
            "exp4_semeval_crosslingual": experiment_4(extractor, model_key),
        }

        # Free GPU memory
        del extractor
        torch.cuda.empty_cache()

    # Save all results
    results_path = RESULTS / "all_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAll results saved to {results_path}")

    return all_results


if __name__ == "__main__":
    main()
