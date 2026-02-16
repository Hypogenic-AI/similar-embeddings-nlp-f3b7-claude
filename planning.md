# Research Plan: Cross-Lingual Embedding Similarity and Polysemy

## Motivation & Novelty Assessment

### Why This Research Matters
Modern multilingual models (mBERT, XLM-R) serve billions of users across 100+ languages. Understanding whether translation equivalents share similar embeddings — and how polysemy disrupts this — is critical for cross-lingual NLP applications (machine translation, cross-lingual information retrieval, multilingual search). If polysemous words with partially overlapping senses still share high similarity, it validates the robustness of multilingual representations; if not, it reveals a systematic weakness that affects downstream applications.

### Gap in Existing Work
Based on our literature review:
- Wu & Conneau (2020) showed cross-lingual structure emerges even without shared vocabulary, but did not directly measure embedding similarity of translation pairs or analyze polysemy effects.
- Zhang et al. (2019) showed polysemy hurts cross-lingual alignment, but only studied static embeddings and supervised alignment, not modern contextual models.
- Upadhyay et al. (2017) proposed multi-sense multilingual embeddings, but pre-dates modern transformers.
- **No systematic study** measures how the number of word senses and sense distribution affects cross-lingual embedding similarity in modern contextual models (mBERT, XLM-R) at both the **type level** (decontextualized) and **token level** (contextualized).

### Our Novel Contribution
We conduct the first systematic comparison of cross-lingual embedding similarity for monosemous vs. polysemous words in modern multilingual transformers, examining:
1. Whether translation equivalents have similar embeddings (type-level and token-level)
2. Whether polysemy degrades this similarity
3. Whether contextualized embeddings (in specific sense contexts) recover the similarity that type-level embeddings lose due to polysemy
4. How these effects vary across language pairs and model layers

### Experiment Justification
- **Experiment 1 (Translation pair similarity)**: Establishes the baseline finding — do translation equivalents have similar embeddings? Uses MUSE dictionaries for large-scale measurement.
- **Experiment 2 (Monosemous vs. polysemous)**: Tests the core polysemy hypothesis — does having multiple senses reduce cross-lingual similarity? Uses WordNet sense counts.
- **Experiment 3 (Contextualized sense-level similarity)**: Tests whether providing sense-disambiguating context recovers similarity. Uses MCL-WiC dataset.
- **Experiment 4 (Cross-lingual word similarity correlation)**: Validates against human judgments on SemEval-2017 Task 2.

## Research Question
Do words with similar meanings in different languages have similar embeddings in multilingual models? Specifically, if a word has multiple meanings and only one meaning is shared across languages, how does this polysemy affect embedding similarity?

## Hypothesis Decomposition
- **H1**: Translation equivalents have significantly higher cosine similarity than random word pairs in multilingual model embeddings.
- **H2**: Monosemous translation pairs have higher cosine similarity than polysemous translation pairs.
- **H3**: For polysemous words, contextualized embeddings in sense-matched contexts have higher similarity than in sense-mismatched contexts.
- **H4**: The polysemy effect varies across model layers, with middle layers showing the best cross-lingual alignment.

## Proposed Methodology

### Approach
We extract embeddings from two multilingual transformer models (mBERT, XLM-R-base) for translation pairs from MUSE bilingual dictionaries. We classify words as monosemous (1 WordNet synset) or polysemous (2+ synsets) and compare embedding similarity distributions. For contextualized analysis, we use the MCL-WiC dataset which provides words in context with same/different sense labels. We also evaluate on SemEval-2017 Task 2 cross-lingual similarity.

### Experimental Steps

1. **Data Preparation**
   - Load MUSE EN-FR, EN-DE, EN-ES, EN-RU, EN-ZH dictionaries
   - Query WordNet for English word sense counts
   - Classify pairs as monosemous (1 sense) vs. polysemous (2+ senses)
   - Sample balanced sets for fair comparison

2. **Experiment 1: Type-level similarity of translation pairs**
   - Extract [CLS] or mean-pooled embeddings for isolated words
   - Compute cosine similarity for all translation pairs
   - Compare against random (non-translation) pairs as control
   - Apply centering (Libovicky et al. 2020) to remove language bias
   - Analyze across layers and models

3. **Experiment 2: Monosemous vs. polysemous comparison**
   - Split translation pairs by polysemy category
   - Compare cosine similarity distributions
   - Statistical test: Mann-Whitney U test (non-parametric)
   - Control for word frequency
   - Analyze by number of senses (1, 2-3, 4-5, 6+)

4. **Experiment 3: Contextualized (sense-level) analysis with MCL-WiC**
   - Extract contextualized embeddings for target words in context
   - Compare similarity for same-sense (T) vs. different-sense (F) pairs
   - Test whether context recovers cross-lingual alignment for polysemous words

5. **Experiment 4: Cross-lingual word similarity (SemEval-2017)**
   - Compute model-based similarity for SemEval word pairs
   - Correlate with human ratings (Spearman)
   - Compare models and layers

### Baselines
- Random word pairs (negative control for H1)
- Monosemous words (positive control for H2)
- Raw vs. centered embeddings (Libovicky et al. 2020)

### Evaluation Metrics
- **Cosine similarity**: Primary metric for embedding alignment
- **Mann-Whitney U test**: For comparing distributions (monosemous vs. polysemous)
- **Cohen's d**: Effect size for practical significance
- **Spearman correlation**: For SemEval-2017 evaluation
- **Accuracy**: For MCL-WiC evaluation

### Statistical Analysis Plan
- Mann-Whitney U test for comparing similarity distributions (non-parametric, appropriate for cosine similarity which may not be normally distributed)
- Significance level: α = 0.05 with Bonferroni correction for multiple comparisons
- Effect sizes via Cohen's d and rank-biserial correlation
- Bootstrap 95% confidence intervals for mean similarities
- Spearman rank correlation for SemEval evaluation

## Expected Outcomes
- **H1 supported**: Translation equivalents should have cosine similarity >> random pairs (expected: 0.5-0.8 vs. 0.0-0.2)
- **H2 supported**: Monosemous pairs should have ~5-15% higher similarity than polysemous pairs
- **H3 supported**: Same-sense contextualized pairs should show higher similarity than different-sense pairs
- **H4**: Middle layers (4-8 out of 12) expected to show best alignment, based on prior work

## Timeline and Milestones
- Phase 2 (Setup): 15 min — environment, install packages, validate data
- Phase 3 (Implementation): 60 min — write experiment scripts
- Phase 4 (Experiments): 60 min — run all experiments
- Phase 5 (Analysis): 30 min — statistical tests and visualizations
- Phase 6 (Documentation): 30 min — REPORT.md and README.md

## Potential Challenges
1. **Subword tokenization**: Multilingual models tokenize words into subword pieces; need mean-pooling strategy
2. **WordNet coverage**: WordNet only covers English; polysemy classification relies on EN side
3. **Language bias in embeddings**: Raw cosine similarity affected by language-specific offsets; centering helps
4. **GPU memory**: Large batch processing needed; 2x RTX 3090 provides ample memory
5. **OOV words**: Some MUSE dictionary words may not be in model vocabulary

## Success Criteria
1. Clear statistical evidence for/against each hypothesis (p-values, effect sizes)
2. Reproducible results across models and language pairs
3. Meaningful visualizations that illustrate findings
4. Complete REPORT.md with actual experimental data
