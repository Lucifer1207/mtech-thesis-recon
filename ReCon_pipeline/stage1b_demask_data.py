"""
stage1b_demask_data.py (v2)
─────────────────────────────────────────────────────────────
Strips RECON_<Type>_ marker prefixes from train.jsonl and
test.jsonl, keeping the actual masked value after the prefix.

No changes needed from original -- just re-run on new Stage 1 output.
"""

import json
import re

RECON_MARKER_PATTERN = re.compile(r'RECON_[A-Za-z]+(?:_[A-Za-z]+)*_')


def demask_text(text):
    return RECON_MARKER_PATTERN.sub('', text)


def process_file(input_path, output_path):
    total = 0
    modified = 0
    marker_counts = {}

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            item = json.loads(line)
            original_text = item['text']

            for match in RECON_MARKER_PATTERN.finditer(original_text):
                marker = match.group()
                marker_counts[marker] = marker_counts.get(marker, 0) + 1

            cleaned_text = demask_text(original_text)
            if cleaned_text != original_text:
                modified += 1

            item['text'] = cleaned_text
            fout.write(json.dumps(item) + "\n")
            total += 1

    return total, modified, marker_counts


def main():
    print("=" * 60)
    print("STAGE 1b (v2): De-masking RECON_* markers")
    print("=" * 60)

    print("\nProcessing train.jsonl...")
    train_total, train_modified, train_markers = process_file(
        "pipeline_data/train.jsonl", "pipeline_data/train_clean.jsonl"
    )
    print(f"  Total: {train_total:,} | Modified: {train_modified:,} "
          f"({100*train_modified/train_total:.2f}%)")

    print("\nProcessing test.jsonl...")
    test_total, test_modified, test_markers = process_file(
        "pipeline_data/test.jsonl", "pipeline_data/test_clean.jsonl"
    )
    print(f"  Total: {test_total:,} | Modified: {test_modified:,} "
          f"({100*test_modified/test_total:.2f}%)")

    all_markers = {}
    for d in [train_markers, test_markers]:
        for k, v in d.items():
            all_markers[k] = all_markers.get(k, 0) + v

    print("\nMarker types removed (sorted by frequency):")
    for marker, count in sorted(all_markers.items(), key=lambda x: -x[1]):
        print(f"  {marker:<35} {count:>6,}")
    print(f"\nTotal distinct types: {len(all_markers)}")
    print(f"Total occurrences removed: {sum(all_markers.values()):,}")

    print("\n--- Verification: checking no markers remain ---")
    for fname in ["pipeline_data/train_clean.jsonl",
                  "pipeline_data/test_clean.jsonl"]:
        remaining = 0
        with open(fname) as f:
            for line in f:
                item = json.loads(line)
                if RECON_MARKER_PATTERN.search(item['text']):
                    remaining += 1
        status = "CLEAN ✅" if remaining == 0 else f"⚠️  {remaining} still have markers!"
        print(f"  {fname}: {status}")

    print("\nSTAGE 1b COMPLETE")


if __name__ == "__main__":
    main()