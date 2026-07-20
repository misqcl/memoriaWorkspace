import os
import shutil
import pandas as pd
import re
from collections import Counter

EXPERIMENT_FOLDER = "Test_ecuestados/Tests"
IMAGE_FOLDER = "images_or"
OUTPUT_FOLDER = "imagenes/top_imagenes"
ALL_IMAGES_FOLDER = "imagenes/filtered_images"

def clean_text(x):
    x = str(x)
    x = x.replace('\ufeff', '')
    x = x.replace('\xa0', ' ')
    x = x.strip().strip('"').strip("'")
    x = re.sub(r'^[A-Z]\.\s*', '', x)
    x = x.strip()
    if x == "" or x == "0":
        return None
    return x

def extract_answers_from_column(column):
    start = None
    end = None

    for i, val in enumerate(column):
        if str(val).strip() == "1 point":
            start = i + 1
            break

    for i in range(start or 0, len(column)):
        if str(column.iloc[i]).strip() == "0":
            end = i
            break

    if start is None or end is None:
        return []

    answers = []

    for cell in column.iloc[start:end]:
        if pd.isna(cell):
            continue

        raw_parts = re.split(r'[\n\r]+', str(cell))

        for part in raw_parts:
            part = part.strip().strip('"').strip("'")
            part = re.sub(r'^[A-Z]\.\s*', '', part)
            part = clean_text(part)
            if part:
                answers.append(part)

    return answers

def extract_number(filename):
    match = re.search(r'Nø(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

def find_low_answer_images(folder):
    files = sorted(
        [f for f in os.listdir(folder) if f.endswith(".xlsx")],
        key=extract_number
    )

    low_images = []
    global_index = 0

    for file in files:
        path = os.path.join(folder, file)
        df = pd.read_excel(path, header=None)

        for col_offset, col_idx in enumerate(range(4, df.shape[1])):
            column = df.iloc[:, col_idx]
            answers = extract_answers_from_column(column)
            count = len(answers)

            image_number = global_index + col_offset

            if count < 5:
                low_images.append(image_number)

        global_index += 100

    return set(low_images)

remove_images = find_low_answer_images(EXPERIMENT_FOLDER)

valid_images = [i for i in range(2000) if i not in remove_images]

print(f"Total originales: 2000")
print(f"Filtradas (eliminadas): {len(remove_images)}")
print(f"Válidas: {len(valid_images)}")

index_mapping = {idx: orig for idx, orig in enumerate(valid_images)}

mapping = {
    'Felicidad': 'Positive',
    'Diversión': 'Positive',
    'Curiosidad': 'Positive',
    'Tristeza': 'Negative',
    'Desagrado': 'Negative',
    'Miedo': 'Negative',
    'Indiferencia': 'Neutral'
}

def extract_data(filepath, start_index, remove_set):
    df = pd.read_excel(filepath, header=None)
    all_items = []

    for col_offset, col_idx in enumerate(range(4, df.shape[1])):

        global_index = start_index + col_offset

        if global_index in remove_set:
            continue

        column = df.iloc[:, col_idx]

        answers = extract_answers_from_column(column)

        if answers:
            all_items.append(answers)

    return all_items

all_data = []
current_index = 0

files = sorted(
    [f for f in os.listdir(EXPERIMENT_FOLDER) if f.endswith(".xlsx")],
    key=extract_number
)

for file in files:
    path = os.path.join(EXPERIMENT_FOLDER, file)

    items = extract_data(
        path,
        start_index=current_index,
        remove_set=remove_images
    )

    all_data.extend(items)
    current_index += 100

mapped_data = [
    [mapping.get(v, v) for v in item if v is not None]
    for item in all_data
]

results = []

for idx, item in enumerate(mapped_data):
    if len(item) < 2:
        continue

    counts = Counter(item)
    dominant_category = counts.most_common(1)[0][0]
    dominance = counts.most_common(1)[0][1] / len(item)

    results.append({
        "clean_index": idx,
        "answers": item,
        "category": dominant_category,
        "dominance": dominance
    })

positive = sorted(
    [r for r in results if r["category"] == "Positive"],
    key=lambda x: (x["dominance"], len(x["answers"])),
    reverse=True
)

negative = sorted(
    [r for r in results if r["category"] == "Negative"],
    key=lambda x: (x["dominance"], len(x["answers"])),
    reverse=True
)

neutral = sorted(
    [r for r in results if r["category"] == "Neutral"],
    key=lambda x: (x["dominance"], len(x["answers"])),
    reverse=True
)

def print_top_AVG(group, name):
    print(f"\nTop 20 {name} images:\n")

    for r in group[:20]:
        print(f"Image: {r['clean_index']+1}")
        print(f"Dominance: {r['dominance']:.4f}")
        print(f"Answers: {r['answers']} {(len(r['answers'])*r['dominance']):.0f}/{len(r['answers'])}")
        print("-" * 100)
print_top_AVG(positive, "Positive")
print_top_AVG(negative, "Negative")
print_top_AVG(neutral,  "Neutral")

categories = {
    "positive": positive[:20],
    "negative": negative[:20],
    "neutral": neutral[:20],
}

for category, items in categories.items():
    folder = os.path.join(OUTPUT_FOLDER, category)
    os.makedirs(folder, exist_ok=True)

    print(f"\nCopying {category} images:")

    for rank, r in enumerate(items, start=1):
        clean_index = r["clean_index"]
        original_index = index_mapping[clean_index]

        src = os.path.join(IMAGE_FOLDER, f"{original_index}.jpg")
        dst = os.path.join(folder, f"{clean_index+1}_orig_{original_index}.jpg")

        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"#{rank} ← clean {clean_index} | orig {original_index}")

os.makedirs(ALL_IMAGES_FOLDER, exist_ok=True)

print("\nCopying all valid images:")

for clean_index, original_index in index_mapping.items():
    src = os.path.join(IMAGE_FOLDER, f"{original_index}.jpg")
    dst = os.path.join(ALL_IMAGES_FOLDER, f"{clean_index+1}.jpg")

    if os.path.exists(src):
        shutil.copy(src, dst)

print("\n✅ Done!")