# Literature Review: Do Words with Similar Meanings in Different Languages Have Similar Embeddings?

## Research Area Overview

This research examines whether words with equivalent meanings across different languages are represented similarly in multilingual embedding models (such as mBERT, XLM-R, and cross-lingual word embeddings). The hypothesis extends to consider how polysemy (words with multiple meanings) affects this similarity: if only one sense of a polysemous word is shared across languages, the overall embedding similarity may be diluted.

The field sits at the intersection of **cross-lingual word embeddings**, **multilingual pretrained language models**, and **word sense disambiguation**. Research over the past decade has established that (1) embedding spaces across languages are approximately isomorphic, (2) multilingual models learn to align translation equivalents even without explicit parallel data, and (3) polysemy complicates this alignment.

---

## Key Papers

### 1. Emerging Cross-lingual Structure in Pretrained Language Models
- **Authors**: Wu & Conneau (2020)
- **Source**: ACL 2020 (arXiv:1911.01464)
- **Key Contribution**: First detailed ablation study showing that cross-lingual representations emerge in multilingual BERT even without shared vocabulary or domain similarity.
- **Methodology**: Trained bilingual masked language models varying shared vocabulary (anchor points), parameter sharing, and domain. Used Procrustes alignment and CKA similarity to compare representations.
- **Key Findings**:
  - Parameter sharing is the most important factor for cross-lingual transfer; shared vocabulary (anchor points) contributes only a few points.
  - Even with zero shared vocabulary, cross-lingual transfer is still effective.
  - Monolingual BERT models trained independently in different languages learn representations that can be aligned via simple linear mappings (Procrustes), achieving decent bilingual dictionary induction.
  - Early transformer layers are more similar across languages than later layers.
  - More closely related languages show stronger alignment effects.
- **Datasets Used**: XNLI, WikiAnn NER, Universal Dependencies parsing, MUSE bilingual dictionaries, Tatoeba sentence retrieval.
- **Relevance**: **Directly supports our hypothesis** - shows words with similar meanings across languages get similar representations even without explicit cross-lingual signals. The "universal latent symmetries" finding is central to our research question.

### 2. A Survey of Cross-lingual Word Embedding Models
- **Authors**: Ruder, Vulic, Sogaard (2019)
- **Source**: JAIR (arXiv:1706.04902)
- **Key Contribution**: Comprehensive typology of cross-lingual word embedding methods, comparing data requirements and objective functions.
- **Methodology**: Survey of mapping-based, pseudo-bilingual, joint methods. Analysis of evaluation approaches.
- **Key Findings**:
  - Many seemingly different CLWE models optimize equivalent objectives.
  - The foundational insight (Mikolov et al., 2013) that word embedding spaces across languages are approximately isomorphic enables linear mapping between them.
  - Evaluation typically uses bilingual lexicon induction (BLI), cross-lingual word similarity, and downstream tasks.
- **Relevance**: Provides theoretical foundation and taxonomy for understanding why translation equivalents cluster together in embedding spaces.

### 3. How Multilingual is Multilingual BERT?
- **Authors**: Pires, Schlinger, Garrette (2019)
- **Source**: ACL 2019 (arXiv:2004.09813)
- **Key Contribution**: Empirical study of mBERT's cross-lingual generalization abilities.
- **Key Findings**:
  - mBERT creates multilingual representations even for language pairs with no common words.
  - Transfer is better for typologically similar languages.
  - Word-piece overlap contributes to but is not necessary for cross-lingual transfer.
- **Relevance**: Confirms that similar-meaning words get similar representations in mBERT.

### 4. Unsupervised Cross-lingual Representation Learning at Scale (XLM-R)
- **Authors**: Conneau et al. (2020)
- **Source**: ACL 2020 (arXiv:1911.02116)
- **Key Contribution**: XLM-R - a multilingual model trained on 100 languages that significantly outperforms mBERT.
- **Key Findings**:
  - Scaling up monolingual data and model capacity improves cross-lingual transfer.
  - Trade-off between positive transfer (shared representations help) and capacity dilution (too many languages hurt).
  - XLM-R is competitive with strong monolingual models, showing multilingual modeling need not sacrifice per-language performance.
- **Relevance**: Establishes XLM-R as the state-of-the-art model for investigating cross-lingual embedding similarity.

### 5. Beyond Bilingual: Multi-sense Word Embeddings using Multilingual Context
- **Authors**: Upadhyay, Chang, Taddy, Kalai, Zou (2017)
- **Source**: RepL4NLP Workshop at ACL 2017
- **Key Contribution**: First multilingual (not just bilingual) approach for learning multi-sense word embeddings.
- **Methodology**: Multi-view Bayesian non-parametric algorithm that uses multilingual parallel corpora to learn different vectors for each sense of a word. Uses English as bridge language.
- **Key Findings**:
  - Different senses of a word may translate to the same word in one language but different words in another (e.g., "interest" maps to same French word but different Chinese words depending on sense).
  - Multilingual training significantly improves sense disambiguation over bilingual training.
  - Using multiple languages helps resolve polysemy that survives in any single translation.
- **Relevance**: **Directly relevant to the polysemy aspect of our hypothesis**. Demonstrates that polysemy is a key challenge for cross-lingual embedding similarity, and multilingual context helps resolve it.

### 6. SemEval-2017 Task 2: Multilingual and Cross-lingual Semantic Word Similarity
- **Authors**: Camacho-Collados, Pilehvar, Collier, Navigli (2017)
- **Source**: SemEval 2017
- **Key Contribution**: High-quality benchmark for multilingual and cross-lingual word similarity across 5 languages (EN, DE, ES, IT, FA).
- **Methodology**: 500 manually curated word pairs per language, covering 34 domains. Cross-lingual datasets for 10 language pairs. Inter-annotator agreement ~0.9.
- **Key Findings**:
  - Systems combining word embeddings with lexical resources perform best.
  - Cross-lingual similarity is harder than monolingual, especially for distant language pairs.
- **Datasets**: Available for direct use in experiments.
- **Relevance**: **Primary evaluation benchmark** for our experiments on cross-lingual word similarity.

### 7. SemEval-2021 Task 2: MCL-WiC (Multilingual and Cross-lingual Word-in-Context Disambiguation)
- **Authors**: Martelli, Kalach, Tola, Navigli (2021)
- **Source**: SemEval 2021
- **Key Contribution**: First manually-annotated dataset for multilingual and cross-lingual word-in-context disambiguation across 5 languages (AR, ZH, EN, FR, RU).
- **Methodology**: Binary classification - determine if a target word in two contexts (same or different language) has the same meaning.
- **Key Findings**:
  - XLM-R based systems achieve best performance.
  - Cross-lingual WiC is harder than monolingual.
  - Covers all open-class parts of speech.
- **Relevance**: **Key dataset for testing the polysemy aspect** of our hypothesis - whether multilingual models can distinguish same vs. different senses across languages.

### 8. Cross-Lingual Contextual Word Embeddings Mapping With Multi-Sense Words In Mind
- **Authors**: Zhang, Yin, Zhu, Zweigenbaum (2019)
- **Source**: EMNLP 2019
- **Key Contribution**: Shows that multi-sense words pose specific challenges for cross-lingual contextual embedding alignment.
- **Methodology**: Proposes noise removal and cluster-level averaging for multi-sense anchors during alignment.
- **Key Findings**:
  - Multi-sense words act as noise in supervised cross-lingual alignment.
  - Removing or replacing multi-sense embeddings during alignment improves bilingual lexicon induction by >10 points for unsupervised methods.
- **Relevance**: **Directly tests our hypothesis about polysemy affecting cross-lingual similarity**. Shows that polysemous words indeed have less similar embeddings across languages.

### 9. On the Language Neutrality of Pre-trained Multilingual Representations
- **Authors**: Libovicky, Rosa, Fraser (2020)
- **Source**: EMNLP 2020 (arXiv:2005.00396)
- **Key Contribution**: Studies whether mBERT representations are truly language-neutral.
- **Key Findings**:
  - mBERT representations are NOT language-neutral - there's a strong language-specific component.
  - After centering (removing language-specific mean), cross-lingual similarity improves significantly.
  - Language identity can be decoded from any layer.
- **Relevance**: Shows that raw cosine similarity between translation equivalents may be affected by language-specific bias, which needs to be accounted for in experiments.

### 10. SensEmBERT: Context-Enhanced Sense Embeddings for Multilingual Word Sense Disambiguation
- **Authors**: Scarlini, Pasini, Navigli (2020)
- **Source**: ACL 2020
- **Key Contribution**: Creates sense embeddings for all WordNet senses using BERT contextualized embeddings, then extends to multilingual setting via BabelNet.
- **Relevance**: Provides methodology for creating sense-specific embeddings from contextual models, useful for testing how individual word senses align cross-lingually.

### 11. Making Monolingual Sentence Embeddings Multilingual Using Knowledge Distillation
- **Authors**: Reimers & Gurevych (2020)
- **Source**: EMNLP 2020 (arXiv:2004.09714)
- **Key Contribution**: Method for extending sentence-BERT to 50+ languages using knowledge distillation.
- **Relevance**: Shows that sentence-level meaning can be aligned across languages, extending the word-level findings.

### 12. First Align, then Predict: Understanding the Cross-Lingual Ability of Multilingual BERT
- **Authors**: Dufter & Schutze (2021)
- **Source**: EACL 2021
- **Key Contribution**: Systematic study of what makes mBERT cross-lingual.
- **Key Findings**:
  - Shared position embeddings and special tokens are sufficient for cross-lingual transfer.
  - The model learns to align same-meaning tokens across languages.
- **Relevance**: Further evidence that meaning similarity drives cross-lingual representation alignment.

---

## Common Methodologies

### Evaluation Approaches
1. **Bilingual Lexicon Induction (BLI)**: Given a source word, retrieve its translation from the target embedding space. Uses CSLS or nearest-neighbor retrieval. Evaluated by Precision@1/5/10. (Used in: Ruder et al., Wu & Conneau, MUSE, VecMap)
2. **Cross-lingual Word Similarity**: Measure correlation between model-predicted similarity and human judgments for word pairs across languages. (Used in: SemEval-2017 Task 2)
3. **Word-in-Context Disambiguation**: Binary classification - do two word occurrences (possibly in different languages) share the same meaning? (Used in: MCL-WiC)
4. **Representation Similarity Analysis**: CKA, SVCCA, or Procrustes alignment to measure structural similarity of embedding spaces. (Used in: Wu & Conneau)

### Alignment Methods
- **Procrustes alignment**: Linear orthogonal mapping between embedding spaces (Mikolov et al., 2013; Smith et al., 2017)
- **Adversarial alignment**: Unsupervised alignment using GANs (Conneau et al., 2017 - MUSE)
- **Joint training**: Multilingual masked language modeling (mBERT, XLM-R)

---

## Standard Baselines

1. **mBERT** (Devlin et al., 2019): Multilingual BERT trained on 104 languages' Wikipedia
2. **XLM-R** (Conneau et al., 2020): Cross-lingual model trained on 100 languages from CommonCrawl
3. **FastText aligned embeddings**: Static word embeddings aligned via MUSE or VecMap
4. **LASER** (Artetxe & Schwenk, 2019): Massively multilingual sentence embeddings

---

## Evaluation Metrics

- **Precision@1/5/10**: For bilingual lexicon induction
- **Spearman/Pearson correlation**: For word similarity tasks
- **Accuracy/F1**: For WiC disambiguation
- **Cosine similarity**: Direct measurement of embedding similarity
- **CKA (Centered Kernel Alignment)**: For structural similarity of representation spaces

---

## Datasets in the Literature

| Dataset | Languages | Task | Used In |
|---------|-----------|------|---------|
| MUSE Dictionaries | 110 language pairs | BLI | Wu & Conneau, many CLWE papers |
| SemEval-2017 Task 2 | EN, DE, ES, IT, FA + cross-lingual pairs | Word similarity | Camacho-Collados et al. |
| MCL-WiC | AR, ZH, EN, FR, RU | Word-in-context disambiguation | Martelli et al. |
| XNLI | 15 languages | NLI transfer | Wu & Conneau, Conneau et al. |
| Tatoeba | 112 languages | Sentence retrieval | Wu & Conneau, Reimers |
| WordSim-353 (translated) | Multiple | Word similarity | Various |
| SimLex-999 (translated) | Multiple | Word similarity | Various |

---

## Gaps and Opportunities

1. **Direct measurement of translation equivalent similarity**: Most papers study cross-lingual transfer on downstream tasks rather than directly measuring embedding similarity of translation pairs. Our research can fill this gap.

2. **Polysemy effects on cross-lingual similarity**: While Zhang et al. (2019) show polysemy hurts alignment, and Upadhyay et al. (2017) show multilingual context helps, no systematic study measures how the number of senses and sense distribution affects cross-lingual embedding similarity in modern models like XLM-R.

3. **Sense-level cross-lingual similarity**: The MCL-WiC dataset enables testing whether specific senses (rather than type-level embeddings) are more aligned cross-lingually.

4. **Language family effects**: The literature consistently shows that closely related languages align better, but the interaction between language distance and polysemy is unexplored.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **MUSE bilingual dictionaries** - For direct measurement of translation equivalent embedding similarity (downloaded)
2. **SemEval-2017 Task 2** - For evaluating graded cross-lingual word similarity (downloaded)
3. **MCL-WiC** - For testing sense-level cross-lingual disambiguation (downloaded)

### Recommended Models
1. **mBERT** (`bert-base-multilingual-cased`) - Widely used baseline
2. **XLM-R** (`xlm-roberta-base` and `xlm-roberta-large`) - State-of-the-art multilingual model
3. **FastText aligned embeddings** (via MUSE) - Static embedding baseline

### Recommended Metrics
1. **Cosine similarity** of translation pair embeddings
2. **Spearman correlation** with human similarity judgments
3. **BLI Precision@1** as a proxy for embedding alignment quality
4. **Accuracy** on MCL-WiC cross-lingual disambiguation

### Experimental Design Suggestions
1. **Experiment 1**: Measure average cosine similarity of translation equivalents from MUSE dictionaries in mBERT/XLM-R embeddings. Compare monosemous vs. polysemous words.
2. **Experiment 2**: Evaluate cross-lingual word similarity on SemEval-2017 Task 2 using different models and layers.
3. **Experiment 3**: Test MCL-WiC cross-lingual disambiguation - analyze whether sense-matched pairs have higher embedding similarity than sense-mismatched pairs.
4. **Experiment 4**: For polysemous words with translations sharing only one sense, measure embedding similarity vs. monosemous translation pairs.

### Methodological Considerations
- Use **mean-pooled subword embeddings** for word-level representations from contextual models.
- Apply **centering** (Libovicky et al., 2020) to remove language-specific bias before computing similarity.
- Compare across **different layers** - early layers may be more language-neutral (Wu & Conneau).
- Control for **word frequency** - rare words may have less stable embeddings.
- Use **WordNet** or **BabelNet** to determine number of senses per word for polysemy analysis.
