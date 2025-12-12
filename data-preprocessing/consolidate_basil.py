"""
BASIL Dataset Consolidation Script

This script combines articles and annotations from the BASIL dataset into a single
consolidated format suitable for training bias detection models.

"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


def split_into_sentences(paragraph_text: str) -> List[str]:
    """
    Split a paragraph into sentences using basic sentence tokenization.

    Args:
        paragraph_text: The paragraph text to split

    Returns:
        List of sentences
    """
    # Use regex to split on sentence boundaries
    # This handles common cases like periods, question marks, and exclamation points
    # while avoiding splits on abbreviations like Mr., Mrs., Dr., etc.
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s')
    sentences = sentence_endings.split(paragraph_text)

    # Clean up sentences and remove empty ones
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def extract_sentences_from_article(body_paragraphs: List[List[str]]) -> List[Dict]:
    """
    Extract all sentences from article body paragraphs with their indices.

    Args:
        body_paragraphs: List of paragraphs, where each paragraph is a list of sentences

    Returns:
        List of dictionaries containing sentence info with para_idx, sent_idx, and text
    """
    sentences = []

    for para_idx, paragraph in enumerate(body_paragraphs):
        # Join all text in the paragraph
        para_text = ' '.join(paragraph)

        # Split into sentences
        para_sentences = split_into_sentences(para_text)

        for sent_idx, sent_text in enumerate(para_sentences):
            sentences.append({
                'para_idx': para_idx,
                'sent_idx': sent_idx,
                'text': sent_text,
                'has_bias': False,
                'annotations': []
            })

    return sentences


def find_phrase_in_sentences(phrase: str, sentences: List[Dict]) -> Optional[int]:
    """
    Find which sentence contains a given phrase.

    Args:
        phrase: The phrase to search for
        sentences: List of sentence dictionaries

    Returns:
        Index of the sentence containing the phrase, or None if not found
    """
    # Normalize the phrase for better matching
    phrase_normalized = phrase.strip()

    for idx, sent_dict in enumerate(sentences):
        sent_text = sent_dict['text']

        # Check if phrase exists in sentence
        if phrase_normalized in sent_text:
            return idx

    # If exact match not found, try fuzzy matching (phrase might span sentences)
    # or might have slight text differences
    for idx, sent_dict in enumerate(sentences):
        sent_text = sent_dict['text']

        # Check if significant portion of phrase is in sentence
        phrase_words = set(phrase_normalized.lower().split())
        sent_words = set(sent_text.lower().split())

        # If more than 70% of phrase words are in the sentence
        if len(phrase_words) > 0:
            overlap = len(phrase_words & sent_words) / len(phrase_words)
            if overlap > 0.7:
                return idx

    return None


def consolidate_article_with_annotations(article_data: Dict, annotation_data: Dict) -> Dict:
    """
    Consolidate article with its annotations.

    Args:
        article_data: Dictionary containing article information
        annotation_data: Dictionary containing annotation information

    Returns:
        Consolidated dictionary in the desired format
    """
    # Extract sentences from article
    sentences = extract_sentences_from_article(article_data['body-paragraphs'])

    # Process phrase-level annotations
    for annotation in annotation_data.get('phrase-level-annotations', []):
        phrase_text = annotation['txt']

        # Find which sentence contains this phrase
        sent_idx = find_phrase_in_sentences(phrase_text, sentences)

        if sent_idx is not None:
            # Mark sentence as having bias
            sentences[sent_idx]['has_bias'] = True

            # Add annotation to the sentence
            sentences[sent_idx]['annotations'].append({
                'txt': annotation['txt'],
                'bias': annotation.get('bias', ''),
                'polarity': annotation.get('polarity', ''),
                'target': annotation.get('target', ''),
                'aim': annotation.get('aim', ''),
                'quote': annotation.get('quote', ''),
                'speaker': annotation.get('speaker', ''),
                'id': annotation.get('id', '')
            })

    # Create consolidated output
    consolidated = {
        'uuid': article_data['uuid'],
        'triplet_uuid': article_data.get('triplet-uuid', ''),
        'article_metadata': {
            'source': article_data.get('source', ''),
            'title': article_data.get('title', ''),
            'date': article_data.get('date', ''),
            'url': article_data.get('url', ''),
            'word_count': article_data.get('word-count', 0),
            'main_entities': article_data.get('main-entities', []),
            'main_event': article_data.get('main-event', '')
        },
        'sentences': sentences,
        'article_level_annotations': annotation_data.get('article-level-annotations', {})
    }

    return consolidated


def process_basil_dataset(basil_dir: str, output_dir: str) -> None:
    """
    Process the entire BASIL dataset and create consolidated files.

    Args:
        basil_dir: Path to the BASIL directory
        output_dir: Path to the output directory for consolidated files
    """
    basil_path = Path(basil_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    articles_dir = basil_path / 'articles'
    annotations_dir = basil_path / 'annotations'

    # Statistics
    total_processed = 0
    total_failed = 0
    failed_files = []

    # Process all years
    for year_dir in sorted(articles_dir.iterdir()):
        if not year_dir.is_dir():
            continue

        year = year_dir.name
        print(f"\nProcessing year {year}...")

        year_output_dir = output_path / year
        year_output_dir.mkdir(exist_ok=True)

        # Process all article files in this year
        for article_file in sorted(year_dir.glob('*.json')):
            # Extract the base name to find corresponding annotation
            # Article file format: {triplet_uuid}_{source_id}.json
            # Annotation file format: {triplet_uuid}_{source_id}_ann.json
            article_basename = article_file.stem  # e.g., "84678e18-0b79-4570-9c9f-cb1cf88d968e_2"
            annotation_file = annotations_dir / year / f"{article_basename}_ann.json"

            if not annotation_file.exists():
                print(f"  Warning: No annotation found for {article_file.name}")
                failed_files.append(str(article_file))
                total_failed += 1
                continue

            try:
                # Load article and annotation
                with open(article_file, 'r', encoding='utf-8') as f:
                    article_data = json.load(f)

                with open(annotation_file, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)

                # Verify UUID match
                if article_data['uuid'] != annotation_data['uuid']:
                    print(f"  Warning: UUID mismatch for {article_file.name}")
                    failed_files.append(str(article_file))
                    total_failed += 1
                    continue

                # Consolidate
                consolidated = consolidate_article_with_annotations(article_data, annotation_data)

                # Save consolidated file
                output_file = year_output_dir / f"{article_basename}_consolidated.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(consolidated, f, indent=2, ensure_ascii=False)

                total_processed += 1

            except Exception as e:
                print(f"  Error processing {article_file.name}: {str(e)}")
                failed_files.append(str(article_file))
                total_failed += 1

    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total files processed successfully: {total_processed}")
    print(f"Total files failed: {total_failed}")

    if failed_files:
        print("\nFailed files:")
        for f in failed_files[:10]:  # Show first 10
            print(f"  - {f}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

    print(f"\nConsolidated files saved to: {output_path}")


def create_single_consolidated_file(basil_dir: str, output_file: str) -> None:
    """
    Create a single consolidated JSON file containing all articles and annotations.

    Args:
        basil_dir: Path to the BASIL directory
        output_file: Path to the output JSON file
    """
    basil_path = Path(basil_dir)
    articles_dir = basil_path / 'articles'
    annotations_dir = basil_path / 'annotations'

    all_data = []
    total_processed = 0
    total_failed = 0

    # Process all years
    for year_dir in sorted(articles_dir.iterdir()):
        if not year_dir.is_dir():
            continue

        year = year_dir.name
        print(f"Processing year {year}...")

        # Process all article files in this year
        for article_file in sorted(year_dir.glob('*.json')):
            article_basename = article_file.stem
            annotation_file = annotations_dir / year / f"{article_basename}_ann.json"

            if not annotation_file.exists():
                total_failed += 1
                continue

            try:
                # Load article and annotation
                with open(article_file, 'r', encoding='utf-8') as f:
                    article_data = json.load(f)

                with open(annotation_file, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)

                # Verify UUID match
                if article_data['uuid'] != annotation_data['uuid']:
                    total_failed += 1
                    continue

                # Consolidate
                consolidated = consolidate_article_with_annotations(article_data, annotation_data)
                all_data.append(consolidated)

                total_processed += 1

            except Exception as e:
                print(f"  Error processing {article_file.name}: {str(e)}")
                total_failed += 1

    # Save all data to single file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total articles processed: {total_processed}")
    print(f"Total articles failed: {total_failed}")
    print(f"\nConsolidated data saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Consolidate BASIL dataset articles and annotations for bias detection model training'
    )
    parser.add_argument(
        '--basil-dir',
        type=str,
        default='./BASIL',
        help='Path to the BASIL directory (default: ./BASIL)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./consolidated',
        help='Path to output directory for consolidated files (default: ./consolidated)'
    )
    parser.add_argument(
        '--single-file',
        type=str,
        help='Create a single consolidated JSON file instead of separate files'
    )

    args = parser.parse_args()

    if args.single_file:
        create_single_consolidated_file(args.basil_dir, args.single_file)
    else:
        process_basil_dataset(args.basil_dir, args.output_dir)


if __name__ == '__main__':
    main()
