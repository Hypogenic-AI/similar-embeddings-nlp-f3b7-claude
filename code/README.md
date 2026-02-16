# Code Repository Index

This directory contains external tools and libraries relevant to the research question:
**"Do words with similar meanings in different languages have similar embeddings in multilingual models? Including polysemy effects."**

---

## 1. MUSE — Multilingual Unsupervised and Supervised Embeddings

**Source:** https://github.com/facebookresearch/MUSE  
**Path:** `code/MUSE/`

### What it provides
- Pre-trained multilingual word embeddings (fastText vectors aligned in a common space)
- 110 large-scale bilingual dictionaries for training and evaluation
- Two alignment methods:
  - **Supervised** — uses a bilingual dictionary or identical character strings
  - **Unsupervised** — requires no parallel data (based on adversarial training + Procrustes refinement)
- Evaluation on word translation, cross-lingual word similarity (SemEval 2017), and sentence translation retrieval (Europarl)

### Key scripts and entry points
| Script | Purpose |
|--------|---------|
| `supervised.py` | Train supervised cross-lingual alignment (Procrustes + CSLS) |
| `unsupervised.py` | Train unsupervised alignment (adversarial + iterative refinement) |
| `evaluate.py` | Evaluate aligned embeddings on word translation, word similarity, sentence retrieval |
| `data/get_evaluation.sh` | Download evaluation datasets (bilingual dictionaries, word similarity, Europarl) |
| `demo.ipynb` | Jupyter notebook demonstrating the pipeline |
| `src/evaluation/word_translation.py` | Word translation evaluation logic |
| `src/evaluation/wordsim.py` | Cross-lingual word similarity evaluation |
| `src/evaluation/sent_translation.py` | Sentence translation retrieval evaluation |
| `src/dico_builder.py` | Dictionary induction from aligned embeddings |
| `src/models.py` | Discriminator and mapping model architectures |
| `src/trainer.py` | Training loop for supervised and unsupervised methods |

### Dependencies
- Python 2/3, NumPy, SciPy
- PyTorch
- Faiss (recommended for fast nearest neighbor search; required for CPU, optional for GPU)

### Relevance to the research
MUSE directly tests whether words with similar meanings across languages end up near each other in a shared embedding space. The supervised method uses known translation pairs to learn a linear mapping; the unsupervised method discovers the alignment without parallel data. Both assume that monolingual embedding spaces are approximately isomorphic -- a key hypothesis underlying the research question. The evaluation tools (word translation accuracy, cross-lingual word similarity) provide quantitative answers to whether translation equivalents share similar embeddings.

---

## 2. VecMap — Cross-Lingual Word Embedding Mappings

**Source:** https://github.com/artetxem/vecmap  
**Path:** `code/vecmap/`

### What it provides
- A framework for learning cross-lingual word embedding mappings with four modes:
  - **Supervised** — requires a large training dictionary
  - **Semi-supervised** — requires a small seed dictionary
  - **Identical** — uses identical words across languages as anchors (no dictionary needed)
  - **Unsupervised** — fully unsupervised, no dictionary or identical words needed
- Evaluation tools for word translation induction, word similarity, and word analogy

### Key scripts and entry points
| Script | Purpose |
|--------|---------|
| `map_embeddings.py` | Main mapping script (all 4 modes: `--supervised`, `--semi_supervised`, `--identical`, `--unsupervised`) |
| `eval_translation.py` | Evaluate bilingual lexicon extraction (nearest neighbor or CSLS retrieval) |
| `eval_similarity.py` | Evaluate cross-lingual word similarity |
| `eval_analogy.py` | Evaluate monolingual word analogies |
| `embeddings.py` | Embedding I/O utilities (word2vec text format) |
| `normalize_embeddings.py` | Normalize embeddings (unit, center, etc.) |
| `get_data.sh` | Download evaluation datasets |

### Example usage
```bash
# Map embeddings (unsupervised)
python3 map_embeddings.py --unsupervised SRC.EMB TRG.EMB SRC_MAPPED.EMB TRG_MAPPED.EMB

# Evaluate word translation with CSLS
python3 eval_translation.py SRC_MAPPED.EMB TRG_MAPPED.EMB -d TEST.DICT --retrieval csls

# Evaluate cross-lingual similarity
python3 eval_similarity.py -l --backoff 0 SRC_MAPPED.EMB TRG_MAPPED.EMB -i TEST_SIMILARITY.TXT
```

### Dependencies
- Python 3, NumPy, SciPy
- CuPy (optional, for CUDA/GPU support)

### Relevance to the research
VecMap provides an alternative (and in some settings, superior) approach to aligning monolingual embedding spaces. Comparing MUSE and VecMap results reveals how robust the "similar meanings = similar embeddings" hypothesis is across different alignment algorithms. The self-learning method (ACL 2018) is particularly relevant since it shows that the structural similarity of embedding spaces is strong enough to discover alignments without any bilingual signal. However, neither MUSE nor VecMap handles polysemy directly -- each word gets a single vector -- making the polysemy analysis a gap these tools help identify.

---

## 3. XLing-Eval — Cross-Lingual Embedding Evaluation

**Source:** https://github.com/codogogo/xling-eval  
**Path:** `code/xling-eval/`

### What it provides
- Bilingual dictionaries for 28 language pairs (training: 500/1K/3K/5K pairs; test: 2K pairs)
- Three cross-lingual mapping methods: PROC (Procrustes), CCA (Canonical Correlation Analysis), PROC-B (bootstrapping extension)
- Standardized BLI (Bilingual Lexicon Induction) evaluation framework
- Based on the ACL 2019 paper: "How to (Properly) Evaluate Cross-Lingual Word Embeddings"

### Key scripts and entry points
| Script | Purpose |
|--------|---------|
| `code/emb_serializer.py` | Convert text embeddings to serialized NumPy format |
| `code/emb_deserializer.py` | Convert serialized embeddings back to text |
| `code/map.py` | Induce bilingual embedding space (PROC, PROC-B, or CCA) |
| `code/eval.py` | Evaluate BLI performance on test dictionaries |
| `code/projection.py` | Projection-based mapping utilities |
| `code/cca.py` | CCA implementation |
| `bli_datasets/` | Bilingual dictionaries for 28 language pairs |

### Dependencies
- Python 3, NumPy, SciPy

### Relevance to the research
This repo provides the most rigorous evaluation framework for cross-lingual embeddings. The accompanying paper highlights common pitfalls in BLI evaluation (e.g., hubness, dictionary quality) that are critical for correctly answering whether translation equivalents have similar embeddings. The 28-language-pair test sets enable broad evaluation across diverse language families.

---

## 4. Wikipedia2Vec — Word and Entity Embeddings from Wikipedia

**Source:** https://github.com/Wikipedia2Vec/wikipedia2vec  
**Path:** `code/wikipedia2vec/`

### What it provides
- Tool for learning embeddings of both words and Wikipedia entities jointly
- Skip-gram model extended to learn entity embeddings
- Pre-trained embeddings available for 12 languages (EN, AR, ZH, NL, FR, DE, IT, JA, PL, PT, RU, ES)
- Can be trained from any Wikipedia dump

### Key entry points
| Component | Purpose |
|-----------|---------|
| `wikipedia2vec train` | Train embeddings from a Wikipedia dump |
| Pre-trained models | Download from wikipedia2vec.github.io for 12 languages |
| `examples/` | Example applications (text classification, etc.) |

### Dependencies
- Python 3, installable via `pip install wikipedia2vec`
- See `requirements.txt` for full list

### Relevance to the research
Wikipedia2Vec is relevant for the polysemy aspect of the research. By learning entity embeddings alongside word embeddings, it provides a way to disambiguate polysemous words (e.g., "bank" the institution vs. "bank" the river bank) through their associated Wikipedia entities. Comparing word-level embeddings across languages (which conflate senses) with entity-level embeddings (which are sense-specific) can reveal how polysemy affects cross-lingual embedding similarity. Pre-trained models for 12 languages enable multilingual comparison out of the box.

---

## How These Tools Work Together

```
Research Pipeline:

1. TRAIN/OBTAIN MONOLINGUAL EMBEDDINGS
   - Use fastText or Wikipedia2Vec to get per-language embeddings

2. ALIGN ACROSS LANGUAGES
   - MUSE (supervised or unsupervised adversarial alignment)
   - VecMap (supervised, semi-supervised, or unsupervised self-learning)
   - XLing-Eval (Procrustes, CCA, or bootstrapped Procrustes)

3. EVALUATE: Do translation equivalents have similar embeddings?
   - Word translation accuracy (MUSE, VecMap, XLing-Eval)
   - Cross-lingual word similarity (MUSE, VecMap)
   - Sentence translation retrieval (MUSE)

4. ANALYZE POLYSEMY EFFECTS
   - Wikipedia2Vec: compare word vs. entity embeddings cross-lingually
   - Identify polysemous words where cross-lingual similarity breaks down
   - Measure whether sense-disambiguated embeddings improve alignment
```

## Quick Start

```bash
# 1. Download MUSE evaluation data
cd code/MUSE/data && bash get_evaluation.sh

# 2. Download VecMap data
cd code/vecmap && bash get_data.sh

# 3. Run MUSE unsupervised alignment (example: EN-ES)
cd code/MUSE
python unsupervised.py --src_lang en --tgt_lang es --src_emb data/wiki.en.vec --tgt_emb data/wiki.es.vec

# 4. Run VecMap unsupervised alignment
cd code/vecmap
python3 map_embeddings.py --unsupervised en.emb es.emb en_mapped.emb es_mapped.emb

# 5. Evaluate with xling-eval
cd code/xling-eval
python3 code/emb_serializer.py en.emb en.vocab en.vectors
python3 code/emb_serializer.py es.emb es.vocab es.vectors
python3 code/map.py -m p -d bli_datasets/en-es/train/en-es.0-5000.txt en.vectors en.vocab es.vectors es.vocab output/
python3 code/eval.py bli_datasets/en-es/test/en-es.test.txt output/en-es.en.vectors output/en-es.es.vectors en.vocab es.vocab
```
