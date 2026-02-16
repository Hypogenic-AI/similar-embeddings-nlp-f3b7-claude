# Datasets

This directory contains the datasets used for the research project:
**"Do words with similar meanings in different languages have similar embeddings in multilingual models?"**

All datasets are excluded from version control via `.gitignore` due to their size.
Use the download instructions below to reproduce.

---

## 1. MUSE Bilingual Dictionaries

**Location:** `muse/`

**Source:** Facebook AI Research (FAIR) — [MUSE: Multilingual Unsupervised and Supervised Embeddings](https://github.com/facebookresearch/MUSE)

**Description:** Ground-truth bilingual dictionaries for bilingual lexicon induction (BLI) evaluation. Each file contains tab-separated word pairs (source -> target).

**Files downloaded:**
| File | Language Pair | Lines (word pairs) |
|------|--------------|-------------------|
| `en-fr.txt` | English -> French | 113,286 |
| `en-de.txt` | English -> German | 101,931 |
| `en-zh.txt` | English -> Chinese | 39,334 |
| `en-ru.txt` | English -> Russian | 53,186 |
| `en-es.txt` | English -> Spanish | 112,580 |

**Total size:** ~4.9 MB

**Format:**
```
the	le
the	les
and	et
```

**Download commands:**
```bash
mkdir -p datasets/muse
cd datasets/muse
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-fr.txt -O en-fr.txt
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-de.txt -O en-de.txt
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-zh.txt -O en-zh.txt
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-ru.txt -O en-ru.txt
wget https://dl.fbaipublicfiles.com/arrival/dictionaries/en-es.txt -O en-es.txt
```

**Use case:** Evaluating whether multilingual embedding models place translation-equivalent words close together in the shared embedding space. These dictionaries serve as ground truth for measuring bilingual lexicon induction accuracy (P@1, P@5, P@10).

---

## 2. MCL-WiC (Multilingual and Cross-lingual Word-in-Context)

**Location:** `mcl-wic/`

**Source:** SemEval-2021 Task 2 — [GitHub](https://github.com/SapienzaNLP/mcl-wic)

**Description:** A benchmark for multilingual and cross-lingual word sense disambiguation in context. Given two sentences containing the same word (or its translation), the task is to determine whether the word is used in the same sense.

**Languages:** English (en), Arabic (ar), French (fr), Russian (ru), Chinese (zh)

**Splits:**
- **Training:** `data/MCL-WiC/training/` — English-English pairs (~8,000 instances in JSON format)
- **Dev:** `data/MCL-WiC/dev/multilingual/` — Monolingual pairs for ar-ar, en-en, fr-fr, ru-ru, zh-zh (~1,000 per language)
- **Test (multilingual):** `data/MCL-WiC/test/multilingual/` — Same-language pairs
- **Test (crosslingual):** `data/MCL-WiC/test/crosslingual/` — Cross-language pairs (en-ar, en-fr, en-ru, en-zh)
- **Gold labels:** `data/test.*.gold` files

**Total size:** ~17 MB (including .git and zip files)

**Format:** JSON with fields: `id`, `lemma`, `pos`, `sentence1`, `sentence2`, `start1`, `end1`, `start2`, `end2`

**Download command:**
```bash
git clone https://github.com/SapienzaNLP/mcl-wic.git datasets/mcl-wic
cd datasets/mcl-wic
unzip SemEval-2021_MCL-WiC_all-datasets.zip -d data/
unzip SemEval-2021_MCL-WiC_test-gold-data.zip -d data/
```

**Use case:** Evaluating whether multilingual models capture not just surface translation equivalence but also fine-grained sense-level similarity across languages. Tests whether embeddings for the same word in the same sense are closer than embeddings for the same word in different senses.

---

## 3. SemEval-2017 Task 2 (Multilingual Word Similarity)

**Location:** `semeval2017-task2/`

**Source:** SemEval-2017 Task 2 — [Official page](http://alt.qcri.org/semeval2017/task2/)

**Description:** Multilingual and cross-lingual word similarity benchmark. Contains word pairs with human-annotated similarity scores (0-4 scale).

**Subtask 1 — Monolingual Word Similarity:**
- 500 word pairs per language
- Languages: English (en), German (de), Spanish (es), Italian (it), Farsi (fa)

**Subtask 2 — Cross-lingual Word Similarity:**
- ~900-978 word pairs per language pair
- 10 language pairs: en-de, en-es, en-fa, en-it, de-es, de-fa, de-it, es-fa, es-it, it-fa

**Includes:** Trial data, test data, gold keys, and evaluation scorer (Java JAR)

**Total size:** ~4.6 MB

**Format:** Tab-separated word pairs (data files) and similarity scores (gold keys)
```
# Data file:
Joule	spacecraft
car	bicycle

# Gold key:
0.78
3.22
```

**Download commands:**
```bash
mkdir -p datasets/semeval2017-task2
cd datasets/semeval2017-task2
wget http://alt.qcri.org/semeval2017/task2/data/uploads/semeval2017-task2.zip
unzip semeval2017-task2.zip
```

**Use case:** Directly measures whether embedding cosine similarity correlates with human judgments of word similarity, both within and across languages. The cross-lingual subtask is particularly relevant — it tests whether "cat" (English) and "gato" (Spanish) receive similar similarity scores to related concepts.

---

## Summary

| Dataset | Task | Languages | Size |
|---------|------|-----------|------|
| MUSE | Bilingual Lexicon Induction | en-fr, en-de, en-zh, en-ru, en-es | 4.9 MB |
| MCL-WiC | Word-in-Context Disambiguation | en, ar, fr, ru, zh (mono + cross) | 17 MB |
| SemEval-2017 Task 2 | Word Similarity | en, de, es, it, fa (mono + cross) | 4.6 MB |

These three datasets complement each other for our research question:
1. **MUSE** tests direct translation equivalence in embedding space (do translation pairs cluster?)
2. **MCL-WiC** tests sense-level similarity (do same-sense translations cluster more than different-sense ones?)
3. **SemEval-2017** tests graded similarity (does embedding distance correlate with human similarity judgments across languages?)
