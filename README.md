# UnbiasedNet

## Overview
**UnbiasedNet** aims to detect and classify media bias at multiple granularities — sentence-level and article-level — by integrating discourse structures, event-relation graphs, and hierarchical aggregation.  

---

## Data
We utilized the [BASIL dataset](https://github.com/launchnlp/BASIL).

---

## TIMELINE (Post Data Preprocessing)

### **1. Baseline Model** -- zero shot, fine tune simple model
**Goal:** Build a baseline classifier for sentence-level and article-level bias.

- Train a model to classify whether a sentence is biased or not.  
- Detect and highlight biased phrases.  
- For article-level bias, treat all sentences equally (uniform aggregation).  
- Record baseline F1 scores at both sentence and article levels.  

**Predicted Improvements Over Baseline:**
- Future models will assign unequal weights to sentences.  
- They will also retain positional information, unlike the baseline.

---

### **2. Sentence-Level Enhancements**
**Goal:** Enrich sentence embeddings with structural and contextual information.

- Add discourse features (e.g., headline, background, consequence).  
- Retain positional embeddings to preserve sentence order.  
- Integrate Event Relation Graph (ERG): -- example graph (one or more)
  - Capture relationships between events (coreference, causal, temporal, subevent).  
    - what type of relationships can be indicative of bias
  - Use a Graph Attention Network (GAT) to refine sentence embeddings with event context.  

---

### **3. Article-Level Aggregation**
**Goal:** Model how sentence-level bias patterns combine to form article-level bias.

#### Hierarchical Attention
- Aggregate sentence embeddings using learned attention weights proportional to bias confidence.  
- Create an article representation vector that emphasizes more biased sentences.  
