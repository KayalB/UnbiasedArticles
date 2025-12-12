# BASIL Dataset Consolidation Summary


### Original BASIL Structure

The BASIL dataset has two main components:

1. **Articles** (`BASIL/articles/{year}/{triplet_uuid}_{source_id}.json`)
   - Contains the article text in `body-paragraphs` (list of paragraphs)
   - Each paragraph is a list of sentences/text chunks
   - Has metadata: title, source (NYT/Fox/HPO), date, main entities, etc.
   - Has a `uuid` field (unique article ID) and `triplet-uuid` (groups 3 articles about same event)

2. **Annotations** (`BASIL/annotations/{year}/{triplet_uuid}_{source_id}_ann.json`)
   - Contains `phrase-level-annotations`: specific biased phrases within the article
   - Contains `article-level-annotations`: overall bias assessments
   - Each phrase annotation has:
     - `txt`: the exact biased phrase
     - `bias`: type ("lex" for lexical, "inf" for informational)
     - `polarity`: sentiment ("pos", "neg", "neu")
     - `target`: the entity being targeted
   - Has matching `uuid` to link to its article

### Matching Logic

Articles and annotations are matched using the **`uuid`** field:
- Each article has a unique UUID
- Each annotation file has the same UUID as its corresponding article
- File naming: article `abc_2.json` → annotation `abc_2_ann.json`

### What the Consolidation Script Does

The `consolidate_basil.py` script combines these two files into a unified format:

1. **Matches** article with annotation using UUID
2. **Extracts sentences** from the article's paragraphs
3. **Maps phrase annotations** to the sentences they appear in
   - Searches for each annotated phrase within the sentences
   - Uses exact matching first, then fuzzy matching (70% word overlap)
4. **Creates consolidated output** with sentence-level structure

### Consolidated Output Format

```json
{
  "uuid": "article-unique-id",
  "triplet_uuid": "event-group-id",
  "article_metadata": {
    "source": "nyt",
    "title": "Article Title",
    "date": "2013-05-22",
    "main_entities": ["Person1", "Person2"],
    "main_event": "Event description"
  },
  "sentences": [
    {
      "para_idx": 0,           // Paragraph index
      "sent_idx": 0,           // Sentence index within paragraph
      "text": "Full sentence text here.",
      "has_bias": true,        // Whether this sentence contains bias
      "annotations": [
        {
          "txt": "biased phrase",
          "bias": "lex",       // Bias type: lex or inf
          "polarity": "neg",   // Sentiment: pos, neg, neu
          "target": "Entity",  // Who/what is targeted
          "aim": "dir",        // Direct or indirect
          "quote": "no",       // Is it a quote?
          "speaker": "",
          "id": "p6"
        }
      ]
    }
  ],
  "article_level_annotations": {
    "author_feeling_Entity": "neg",
    "relative_stance": "left",
    ...
  }
}
```
