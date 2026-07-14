import json
from pathlib import Path
from collections import defaultdict

# Root directory
ROOT_DIR = Path("data")

# Dictionary to store stats per folder: {folder_path: {'apk': 0, 'flow': 0, 'pii': 0, 'non_pii': 0}}
stats = defaultdict(lambda: {'apk': 0, 'flow': 0, 'pii': 0, 'non_pii': 0})

print("Scanning all directories and categorizing by source...\n")

for json_file in ROOT_DIR.rglob("*.json"):
    # Get the parent directory name for categorization
    folder_name = str(json_file.parent)
    
    try:
        with open(json_file, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)

        packets = data.values() if isinstance(data, dict) else data
        
        # We only count this as an APK/file if it contains at least one android flow
        found_android = False
        
        for packet in packets:
            if isinstance(packet, dict) and packet.get("platform") == "android":
                found_android = True
                stats[folder_name]['flow'] += 1
                
                label = int(packet.get("label", 0))
                if label == 1:
                    stats[folder_name]['pii'] += 1
                else:
                    stats[folder_name]['non_pii'] += 1
        
        if found_android:
            stats[folder_name]['apk'] += 1

    except Exception as e:
        print(f"Error reading {json_file}: {e}")

# Print the formatted breakdown
print(f"{'Folder Path':<30} | {'APKs':<6} | {'Total':<8} | {'PII':<6} | {'Non-PII':<8}")
print("-" * 75)

total_apk = total_flow = total_pii = total_non_pii = 0

for folder, s in sorted(stats.items()):
    print(f"{folder:<30} | {s['apk']:<6,} | {s['flow']:<8,} | {s['pii']:<6,} | {s['non_pii']:<8,}")
    total_apk += s['apk']
    total_flow += s['flow']
    total_pii += s['pii']
    total_non_pii += s['non_pii']

print("-" * 75)
print(f"{'GRAND TOTAL':<30} | {total_apk:<6,} | {total_flow:<8,} | {total_pii:<6,} | {total_non_pii:<8,}")
