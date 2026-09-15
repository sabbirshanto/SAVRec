
# %% PUBLIC NOTEBOOK CELL 1
# ==============================================================================
# CELL 1: SETUP, GOOGLE DRIVE MOUNT & KAGGLE AUTHENTICATION
# ==============================================================================
import os
import glob
import shutil
import subprocess
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from google.colab import drive, files

# 1. Safely mount Google Drive
try:
    drive.mount('/content/drive', force_remount=False)
except Exception as e:
    print(f"Drive already connected or handled: {e}")

# 2. Define permanent project directory
DRIVE_PATH = os.environ.get(
    'SAVREC_DATA_ROOT',
    '/content/drive/MyDrive/MS_new/Thesis_HotelRec_Data'
)
os.makedirs(DRIVE_PATH, exist_ok=True)
PARQUET_FILE = os.path.join(DRIVE_PATH, 'merged_hotelrec.parquet')

# 3. Configure Kaggle API Token
if not os.path.exists('/root/.kaggle/kaggle.json'):
    print("📤 Please upload your 'kaggle.json' API token file:")
    uploaded = files.upload()
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    shutil.move('kaggle.json', os.path.expanduser('~/.kaggle/kaggle.json'))
    os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
    print("✅ Kaggle token configured successfully!")
else:
    print("✅ Kaggle API token already configured.")

# %% PUBLIC NOTEBOOK CELL 2
# ==============================================================================
# CELL 2: ZERO-RAM KAGGLE CHUNK PARSER & COMPILER
# ==============================================================================
LOCAL_PARQUET = '/content/temp_merged.parquet'

# 1. Auto-discover or download dataset files
file_paths = sorted(glob.glob('./dataset/hotelrec/**/*.*', recursive=True)) + sorted(glob.glob('./**/*.csv', recursive=True))
file_paths = [f for f in file_paths if 'temp_merged' not in f and 'hotel_info' not in f]

if len(file_paths) == 0 and not os.path.exists(PARQUET_FILE):
    print("📥 Downloading HotelRec splits from Kaggle...")
    os.makedirs('./dataset/hotelrec', exist_ok=True)
    for dataset_name in [
        'hariwh0/hotelrec-dataset-1',
        'hariwh0/hotelrec-dataset-2',
        'hariwh0/hotelrec-dataset-3',
        'hariwh0/hotelrec-dataset-4',
    ]:
        subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', './dataset/hotelrec', '--unzip'],
            check=True,
        )
    file_paths = sorted(glob.glob('./dataset/hotelrec/**/*.*', recursive=True))

# 2. Stream-compile to disk if master parquet doesn't exist yet
if not os.path.exists(PARQUET_FILE):
    print(f"📁 Found {len(file_paths)} raw data files. Streaming directly to disk...")
    col_mapping = {
        'userId': 'user_id', 'user': 'user_id', 'reviewerID': 'user_id', 'author': 'user_id',
        'hotelId': 'hotel_id', 'item': 'hotel_id', 'itemID': 'hotel_id', 'hotel_url': 'hotel_id', 'hotel': 'hotel_id',
        'overall': 'rating', 'score': 'rating', 'stars': 'rating', 'rating': 'rating',
        'text': 'review_text', 'review': 'review_text', 'content': 'review_text', 'review_text': 'review_text',
        'date': 'timestamp', 'reviewTime': 'timestamp', 'timestamp': 'timestamp', 'time': 'timestamp'
    }
    STANDARD_COLUMNS = ['user_id', 'hotel_id', 'rating', 'review_text', 'timestamp']

    writer = None
    total_rows = 0

    for file_path in file_paths:
        if not (file_path.endswith('.csv') or file_path.endswith('.json')):
            continue
        print(f"   -> Processing {os.path.basename(file_path)}...")
        try:
            chunk_container = pd.read_json(file_path, lines=True, chunksize=100000) if file_path.endswith('.json') else pd.read_csv(file_path, chunksize=100000, low_memory=False)
            for chunk in chunk_container:
                chunk.rename(columns=col_mapping, inplace=True)
                chunk = chunk[[c for c in STANDARD_COLUMNS if c in chunk.columns]]
                chunk.dropna(subset=[c for c in ['user_id', 'hotel_id'] if c in chunk.columns], inplace=True)

                if 'user_id' in chunk.columns: chunk['user_id'] = chunk['user_id'].astype(str)
                if 'hotel_id' in chunk.columns: chunk['hotel_id'] = chunk['hotel_id'].astype(str)
                if 'rating' in chunk.columns: chunk['rating'] = pd.to_numeric(chunk['rating'], errors='coerce').astype('float32')
                if 'review_text' in chunk.columns: chunk['review_text'] = chunk['review_text'].fillna('').astype(str)
                if 'timestamp' in chunk.columns: chunk['timestamp'] = chunk['timestamp'].astype(str)

                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(LOCAL_PARQUET, table.schema, compression='snappy')
                writer.write_table(table)
                total_rows += len(chunk)
        except Exception as e:
            print(f"⚠️ Skipped {file_path}: {e}")

    if writer:
        writer.close()
        print(f"\n🎉 Successfully compiled {total_rows:,} rows! Copying to Drive...")
        shutil.copy2(LOCAL_PARQUET, PARQUET_FILE)
        os.remove(LOCAL_PARQUET)
        print(f"✅ Master dataset saved at: {PARQUET_FILE}")
else:
    print(f"✅ Master dataset already exists in Drive: {PARQUET_FILE}")

# %% PUBLIC NOTEBOOK CELL 3
# ==============================================================================
# CELL 3 (v3 FIXED): STRICT 1:1 PROPERTY ALIGNMENT WITH CORRECT HEADERS
# ==============================================================================
import re
import os
import gc
import pandas as pd
import pyarrow.parquet as pq
from google.colab import files

# ------------------------------------------------------------------ CONFIG ---
DRIVE_PATH = os.environ.get(
    'SAVREC_DATA_ROOT',
    '/content/drive/MyDrive/MS_new/Thesis_HotelRec_Data'
)
PARQUET_FILE     = os.path.join(DRIVE_PATH, 'merged_hotelrec.parquet')
REGISTRY_OUT     = os.path.join(DRIVE_PATH, 'item_images_v3.parquet')
INTERACTIONS_OUT = os.path.join(DRIVE_PATH, 'multimodal_interactions_v3.parquet')
AUDIT_OUT        = os.path.join(DRIVE_PATH, 'match_audit_v3.parquet')

MIN_TOKENS = 2   # a 1-token normalized name is a brand, not a property

FILLER = {'hotel', 'motel', 'resort', 'suites', 'suite', 'inn',
          'by', 'the', 'of', 'at', 'llc', 'inc', 'lodge'}

def normalize_hotel_string(name):
    """Lowercase, strip punctuation and generic hospitality words. Keeps digits."""
    if not isinstance(name, str):
        return ""
    name = name.lower().replace('&', ' and ').replace('st.', 'saint').replace('/', ' ')
    name = re.sub(r'[^a-z0-9\s]', ' ', name)
    toks = [w for w in name.split()
            if w not in FILLER and (len(w) > 1 or w.isdigit())]
    return " ".join(toks)

def parse_ta_slug(url):
    s = str(url)
    if '-Reviews-' not in s:
        return "", ""
    tail = re.sub(r'\.html?$', '', s.split('-Reviews-', 1)[1])
    parts = tail.split('-')
    name = parts[0].replace('_', ' ')
    loc = parts[-1].replace('_', ' ') if len(parts) > 1 else ""
    return name, loc

# -------------------------------------------------------------------- LOAD ---
print("Loading interaction columns...")
available_cols = pq.read_schema(PARQUET_FILE).names
cols_to_load = [c for c in ['user_id', 'hotel_id', 'rating', 'timestamp'] if c in available_cols]
df = pd.read_parquet(PARQUET_FILE, columns=cols_to_load)
df['hotel_id'] = df['hotel_id'].astype(str).str.strip()

print("\n📤 Please upload 'hotel_info.csv', 'train_set.csv' AND 'test_set.csv':")
uploaded = files.upload()

# Separate files dynamically based on filenames
meta_filename = [k for k in uploaded if 'info' in k.lower() or 'meta' in k.lower()][0]
train_filename = [k for k in uploaded if 'train' in k.lower()][0]
test_filename = [k for k in uploaded if 'test' in k.lower()][0]

h50k = pd.read_csv(meta_filename)
h50k.rename(columns={'id': 'hotel_id'}, inplace=True)
h50k['hotel_id'] = h50k['hotel_id'].astype(str).str.split('.').str.get(0).str.strip()

# Correctly load train (NO header) and test (WITH header)
print("⚙️ Loading and formatting Hotels-50K image splits...")
train_img_df = pd.read_csv(train_filename, header=None, names=['image_id', 'hotel_id', 'image_url', 'image_source', 'upload_timestamp'])
test_img_df = pd.read_csv(test_filename)

img_df = pd.concat([train_img_df, test_img_df], ignore_index=True)
img_df['hotel_id'] = img_df['hotel_id'].astype(str).str.split('.').str.get(0).str.strip()

del uploaded, train_img_df, test_img_df
gc.collect()

print(f"📊 Total loaded image rows: {len(img_df):,} across {img_df['hotel_id'].nunique():,} unique hotel IDs.")

# ------------------------------------------------------- NORMALIZE BOTH SIDES ---
print("\nNormalizing names...")
ta = pd.DataFrame({'ta_hotel_id': df['hotel_id'].unique()})
parsed = ta['ta_hotel_id'].apply(parse_ta_slug)
ta['ta_name'] = parsed.str[0]
ta['ta_loc'] = parsed.str[1]
ta['core_name'] = ta['ta_name'].apply(normalize_hotel_string)

h50k['core_name'] = h50k['hotel_name'].astype(str).apply(normalize_hotel_string)

# ------------------------------------------------------- STRICT 1:1 MATCHING ---
ta_collapse = ta.groupby('core_name')['ta_hotel_id'].nunique()
h50k_collapse = h50k.groupby('core_name')['hotel_id'].nunique()

ta_unique = set(ta_collapse[ta_collapse == 1].index)
h50k_unique = set(h50k_collapse[h50k_collapse == 1].index)
shared = ta_unique & h50k_unique

pairs = (ta[ta['core_name'].isin(shared)]
         .merge(h50k[h50k['core_name'].isin(shared)][['core_name', 'hotel_id', 'hotel_name']],
                on='core_name', how='inner')
         .rename(columns={'hotel_id': 'img_hotel_id'}))

print(f"\nUnambiguous on both sides: {len(shared):,} names -> {len(pairs):,} candidate pairs")

# ------------------------------------------------- POOL IMAGES BY hotel_id ----
imgs_by_id = img_df.groupby('hotel_id')['image_url'].apply(list)

pairs['available_images'] = pairs['img_hotel_id'].map(imgs_by_id)
pairs['available_images'] = pairs['available_images'].apply(lambda x: x if isinstance(x, list) else [])
pairs['image_count'] = pairs['available_images'].str.len()

registry = pairs[pairs['image_count'] > 0][
    ['ta_hotel_id', 'img_hotel_id', 'core_name', 'ta_name', 'hotel_name', 'available_images', 'image_count']
].copy()

# ------------------------------------------------------------- ASSERTIONS ----
assert registry['ta_hotel_id'].nunique() == len(registry), "TripAdvisor IDs not 1:1"
assert registry['img_hotel_id'].nunique() == len(registry), "Hotels-50K IDs not 1:1"
print("Assertions passed: alignment is strictly 1:1.")

# -------------------------------------------------- LINK TO INTERACTIONS -----
matched = df.merge(
    registry[['ta_hotel_id', 'img_hotel_id', 'core_name']],
    left_on='hotel_id', right_on='ta_hotel_id', how='inner'
).drop(columns=['ta_hotel_id'])

registry.to_parquet(REGISTRY_OUT, index=False)
matched.to_parquet(INTERACTIONS_OUT, index=False, compression='snappy')
registry.drop(columns=['available_images']).to_parquet(AUDIT_OUT, index=False)

# ------------------------------------------------------------------ FUNNEL ---
print("\n" + "=" * 66)
print("ALIGNMENT FUNNEL (FIXED)")
print("=" * 66)
print(f"TripAdvisor properties in reviews : {len(ta):,}")
print(f"  unambiguous on both sides       : {len(shared):,}")
print(f"  with >=1 pooled image           : {len(registry):,}")
print("-" * 66)
print(f"Interactions retained             : {len(matched):,} ({len(matched)/len(df)*100:.2f}% of {len(df):,})")
print(f"Total images                      : {registry['image_count'].sum():,}")
print(f"Images per hotel  median/mean/max : {registry['image_count'].median():.0f} / {registry['image_count'].mean():.1f} / {registry['image_count'].max():,}")
print("=" * 66)
print(f"\nSaved successfully to Drive!")

del df, img_df, h50k
gc.collect()

# %% PUBLIC NOTEBOOK CELL 5
# ==============================================================================
# CELL 5 (v3): HYBRID STRATIFIED SPLITTER (OPTIMIZED FOR SPARSE USERS)
# ==============================================================================
import os
import numpy as np
import pandas as pd

DRIVE_PATH = os.environ.get(
    'SAVREC_DATA_ROOT',
    '/content/drive/MyDrive/MS_new/Thesis_HotelRec_Data'
)
INTERACTIONS_FILE = os.path.join(DRIVE_PATH, 'multimodal_interactions_v3.parquet')

print("⚡ Loading v3 interactions for splitting...")
df = pd.read_parquet(INTERACTIONS_FILE)

# 1. Implicit feedback filter (Rating >= 4.0)
print("⚙️ Filtering for positive implicit feedback (Rating >= 4.0)...")
df_implicit = df[df['rating'] >= 4.0].copy()
df_implicit['label'] = 1

# 2. Item-centric density filter (Ensure every hotel has at least 5 reviews)
K_CORE_ITEM = 5
item_counts = df_implicit['hotel_id'].value_counts()
valid_items = item_counts[item_counts >= K_CORE_ITEM].index
df_implicit = df_implicit[df_implicit['hotel_id'].isin(valid_items)]

print(f"📊 Filtered items (>= {K_CORE_ITEM} reviews): {df_implicit['hotel_id'].nunique():,} hotels remaining.")
print(f"📊 Total interactions entering split pipeline: {len(df_implicit):,}")

# 3. Hybrid User-Stratified 8:1:1 Split
print("\n⚡ Executing Hybrid Stratified Split...")
np.random.seed(42)

u_counts = df_implicit['user_id'].value_counts()
dense_users = u_counts[u_counts >= 3].index
sparse_users = u_counts[u_counts < 3].index

df_dense = df_implicit[df_implicit['user_id'].isin(dense_users)]
df_sparse = df_implicit[df_implicit['user_id'].isin(sparse_users)]

def stratified_811_split(group):
    indices = group.index.values
    np.random.shuffle(indices)
    n = len(indices)
    n_val = max(1, int(np.floor(n * 0.10)))
    n_test = max(1, int(np.floor(n * 0.10)))
    n_train = n - n_val - n_test
    if n_train < 1: n_test, n_val, n_train = 1, 1, n - 2
    return pd.Series({
        'train': indices[:n_train],
        'val': indices[n_train:n_train + n_val],
        'test': indices[n_train + n_val:]
    })

splits = df_dense.groupby('user_id', group_keys=False).apply(stratified_811_split)

# Combine dense user training edges with all sparse user edges (which go strictly to train)
train_df = pd.concat([
    df_dense.loc[np.concatenate(splits['train'].values)],
    df_sparse
]).reset_index(drop=True)

val_df = df_dense.loc[np.concatenate(splits['val'].values)].reset_index(drop=True)
test_df = df_dense.loc[np.concatenate(splits['test'].values)].reset_index(drop=True)

print(f"\n🎯 FINAL EVALUATION SPLITS:")
print(f"   Train Set : {len(train_df):,} interactions")
print(f"   Val Set   : {len(val_df):,} interactions")
print(f"   Test Set  : {len(test_df):,} interactions")

# 4. Save splits permanently to Drive
train_df.to_parquet(os.path.join(DRIVE_PATH, 'train_v3.parquet'), index=False)
val_df.to_parquet(os.path.join(DRIVE_PATH, 'val_v3.parquet'), index=False)
test_df.to_parquet(os.path.join(DRIVE_PATH, 'test_v3.parquet'), index=False)

print(f"\n✅ All v3 splits saved successfully to Drive!")

# %% PUBLIC NOTEBOOK CELL 7
# ==============================================================================
# CELL 6 (v3): THREADED RESNET-50 VISUAL FEATURE EXTRACTOR
#
#   - 16 parallel download threads, batched GPU inference
#   - Chunked checkpointing (resume-safe across Colab disconnects)
#   - Per-hotel coverage manifest: how many images actually embedded, and why
#     the rest failed  -> required for the Limitations section
#   - Raw 2048-d pooled features, no untrained projection
#   - Validity gates: HTTP status, Content-Type, minimum size
# ==============================================================================
import os
import gc
import threading
from io import BytesIO
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms

# ------------------------------------------------------------------ CONFIG ---
DRIVE_PATH = os.environ.get(
    'SAVREC_DATA_ROOT',
    '/content/drive/MyDrive/MS_new/Thesis_HotelRec_Data'
)
REGISTRY_FILE  = os.path.join(DRIVE_PATH, 'item_images_v3.parquet')
TRAIN_FILE     = os.path.join(DRIVE_PATH, 'train_v3.parquet')
EMBEDDINGS_OUT = os.path.join(DRIVE_PATH, 'item_visual_embeddings_v3.pt')
MANIFEST_OUT   = os.path.join(DRIVE_PATH, 'visual_coverage_v3.csv')

CHUNK_HOTELS = 200    # checkpoint granularity
MAX_WORKERS  = 16     # parallel downloads
GPU_BATCH    = 64
TIMEOUT      = 8      # generous: a short timeout mislabels slow links as dead
MIN_PIXELS   = 32     # reject 1x1 trackers / placeholder stubs

torch.manual_seed(42)
np.random.seed(42)

# -------------------------------------------------------------------- SETUP ---
print("Loading registry...")
registry_df = pd.read_parquet(REGISTRY_FILE)
registry_df['img_hotel_id'] = registry_df['img_hotel_id'].astype(str)

# Only extract for items that survived the 5-core filter and are actually used
if os.path.exists(TRAIN_FILE):
    keep = set(pd.read_parquet(TRAIN_FILE)['img_hotel_id'].astype(str))
    before = len(registry_df)
    registry_df = registry_df[registry_df['img_hotel_id'].isin(keep)]
    print(f"  restricted to items present in train: {len(registry_df):,} of {before:,}")

total_urls = int(registry_df['image_count'].sum())
print(f"  {len(registry_df):,} hotels / {total_urls:,} image URLs")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  device: {device}")

weights = models.ResNet50_Weights.DEFAULT
resnet = models.resnet50(weights=weights)
feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ------------------------------------------------------ THREAD-LOCAL SESSION ---
_local = threading.local()


def _session():
    if not hasattr(_local, 's'):
        s = requests.Session()
        s.headers.update({'User-Agent': 'Mozilla/5.0 (academic research)'})
        _local.s = s
    return _local.s


def fetch(task):
    """(hotel_id, url) -> (hotel_id, tensor_or_None, reason)"""
    hid, url = task
    try:
        r = _session().get(url, timeout=TIMEOUT)
        if r.status_code != 200:
            return hid, None, f'http_{r.status_code}'
        if not r.headers.get('Content-Type', '').lower().startswith('image/'):
            return hid, None, 'not_image'
        img = Image.open(BytesIO(r.content)).convert('RGB')
        if min(img.size) < MIN_PIXELS:
            return hid, None, 'too_small'
        return hid, transform(img), 'ok'
    except Exception as e:
        return hid, None, type(e).__name__


# ---------------------------------------------------------------- RESUME ------
item_embeddings = {}
done = set()

if os.path.exists(MANIFEST_OUT):
    prev = pd.read_csv(MANIFEST_OUT)
    done = set(prev['img_hotel_id'].astype(str))
    print(f"Resuming: {len(done):,} hotels already done")
    if os.path.exists(EMBEDDINGS_OUT):
        item_embeddings = torch.load(EMBEDDINGS_OUT, weights_only=False)

pending = registry_df[~registry_df['img_hotel_id'].isin(done)]
print(f"Remaining: {len(pending):,} hotels\n")

# ----------------------------------------------------------------- EXTRACT ----
from concurrent.futures import ThreadPoolExecutor

fail_reasons = defaultdict(int)
chunks = [pending.iloc[i:i + CHUNK_HOTELS]
          for i in range(0, len(pending), CHUNK_HOTELS)]

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    for chunk in tqdm(chunks, desc="chunks"):

        tasks = []
        for _, r in chunk.iterrows():
            hid = str(r['img_hotel_id'])
            imgs = r['available_images']
            if imgs is not None and len(imgs) > 0:
                for u in imgs:
                    tasks.append((hid, u))
        n_urls = defaultdict(int)
        for hid, _ in tasks:
            n_urls[hid] += 1

        per_hotel = defaultdict(list)
        buf_t, buf_h = [], []

        def flush():
            if not buf_t:
                return
            with torch.no_grad():
                out = feature_extractor(torch.stack(buf_t).to(device))
            out = out.squeeze(-1).squeeze(-1).cpu().numpy().astype(np.float32)
            for j, h in enumerate(buf_h):
                per_hotel[h].append(out[j])
            buf_t.clear()
            buf_h.clear()

        for hid, tensor, reason in pool.map(fetch, tasks):
            if tensor is None:
                fail_reasons[reason] += 1
                continue
            buf_t.append(tensor)
            buf_h.append(hid)
            if len(buf_t) >= GPU_BATCH:
                flush()
        flush()

        # ---- aggregate to one vector per hotel, record coverage --------------
        rows = []
        for _, r in chunk.iterrows():
            hid = str(r['img_hotel_id'])
            feats = per_hotel.get(hid, [])
            if feats:
                item_embeddings[hid] = np.mean(feats, axis=0).astype(np.float32)
            rows.append({'img_hotel_id': hid,
                         'n_urls': n_urls.get(hid, 0),
                         'n_embedded': len(feats),
                         'has_visual': int(len(feats) > 0)})

        pd.DataFrame(rows).to_csv(MANIFEST_OUT, mode='a', index=False,
                                  header=not os.path.exists(MANIFEST_OUT))
        torch.save(item_embeddings, EMBEDDINGS_OUT)

# ------------------------------------------------------------------ REPORT ----
man = pd.read_csv(MANIFEST_OUT)
man = man.drop_duplicates(subset='img_hotel_id', keep='last')
covered = man['has_visual'].sum()

print("\n" + "=" * 66)
print("VISUAL COVERAGE")
print("=" * 66)
print(f"Hotels with >=1 embedded image : {covered:,} / {len(man):,} "
      f"({covered / len(man) * 100:.2f}%)")
print(f"Hotels with NO usable image    : {len(man) - covered:,}  <- exclude "
      f"from image-conditioned models")
print(f"URLs attempted / embedded      : {man['n_urls'].sum():,} / "
      f"{man['n_embedded'].sum():,} "
      f"({man['n_embedded'].sum() / max(man['n_urls'].sum(), 1) * 100:.2f}%)")
ok = man[man['has_visual'] == 1]['n_embedded']
if len(ok):
    print(f"Images per covered hotel       : median {ok.median():.0f} | "
          f"mean {ok.mean():.1f} | max {ok.max():,}")
print("\nFailure reasons (this run):")
for k, v in sorted(fail_reasons.items(), key=lambda x: -x[1])[:10]:
    print(f"  {k:<24} {v:,}")
print("=" * 66)
print(f"\nSaved:\n  {EMBEDDINGS_OUT}\n  {MANIFEST_OUT}")
print("\nNOTE: features are raw 2048-d and NOT normalised. Fit any scaling or "
      "PCA on TRAIN items only (§1.6).")

del feature_extractor, resnet
torch.cuda.empty_cache()
gc.collect()

# %% PUBLIC NOTEBOOK CELL 9
# ==============================================================================
# CELL 7 (v6): CLEAN LEAK-FREE SPLIT
#
# FIXES THE V5 USER-HOTEL DUPLICATION / CROSS-SPLIT LEAKAGE
#
# Key rule:
#   One unique (user_id, img_hotel_id) interaction is created BEFORE splitting.
#
# If a user reviewed the same hotel multiple times:
#   - all those reviews remain associated with the same interaction
#   - review_text is concatenated
#   - the pair is assigned to ONLY ONE split
#
# The train/validation/test protocol remains:
#   >=2 unique user-hotel interactions -> eligible for evaluation
#   exactly 2 -> 1 train + 1 validation, no test
#   >=3 -> approximately 8:1:1 train/validation/test
#   <2  -> train only
#
# Output:
#   train_v6.parquet
#   val_v6.parquet
#   test_v6.parquet
#   train_augmented_v6.parquet
#   evaluation_catalog_items_v6.parquet
#   train_history_v6.parquet
#   user_map_v6.json
#   item_map_v6.json
# ==============================================================================

import os
import json
import gc
import numpy as np
import pandas as pd


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

DRIVE_PATH = os.environ.get(
    'SAVREC_DATA_ROOT',
    '/content/drive/MyDrive/MS_new/Thesis_HotelRec_Data'
)

SOURCE_FILE = os.path.join(
    DRIVE_PATH,
    'multimodal_interactions_v3.parquet'
)

VISUAL_MANIFEST = os.path.join(
    DRIVE_PATH,
    'visual_coverage_v3.csv'
)

SEED = 42

K_CORE_ITEM = 5

DENSE_THRESHOLD = 2


print("=" * 85)
print("BUILDING V6 — CLEAN LEAK-FREE USER-HOTEL SPLIT")
print("=" * 85)


# ==============================================================================
# 2. LOAD ORIGINAL INTERACTION DATA
# ==============================================================================

print("\n[1] Loading original interaction data...")

df = pd.read_parquet(
    SOURCE_FILE
)

print(
    f"Original rows : {len(df):,}"
)

print(
    f"Original users: {df['user_id'].nunique():,}"
)

print(
    f"Original hotels: {df['hotel_id'].nunique():,}"
)


# ==============================================================================
# 3. NORMALIZE IDENTIFIERS
# ==============================================================================

df['user_id'] = (
    df['user_id']
    .astype(str)
    .str.strip()
)

df['hotel_id'] = (
    df['hotel_id']
    .astype(str)
    .str.strip()
)

df['img_hotel_id'] = (
    df['img_hotel_id']
    .astype(str)
    .str.strip()
)


# ==============================================================================
# 4. POSITIVE IMPLICIT FEEDBACK
# ==============================================================================

print(
    "\n[2] Filtering positive implicit interactions "
    "(rating >= 4.0)..."
)

df_implicit = df[
    df['rating'] >= 4.0
].copy()

df_implicit['label'] = 1

print(
    f"Positive rows : "
    f"{len(df_implicit):,}"
)


# ==============================================================================
# 5. KEEP ONLY VISUALLY COVERED HOTELS
# ==============================================================================

print(
    "\n[3] Applying visual-coverage filter..."
)

manifest = pd.read_csv(
    VISUAL_MANIFEST
)

covered_items = set(
    manifest.loc[
        manifest['has_visual'] == 1,
        'img_hotel_id'
    ]
    .astype(str)
)

df_implicit = df_implicit[
    df_implicit['img_hotel_id']
    .isin(covered_items)
].copy()

print(
    f"Visually covered hotels : "
    f"{len(covered_items):,}"
)

print(
    f"Rows after image filter  : "
    f"{len(df_implicit):,}"
)


# ==============================================================================
# 6. IMPORTANT: CREATE ONE INTERACTION PER USER-HOTEL
# ==============================================================================
#
# THIS IS THE CRITICAL FIX.
#
# Multiple reviews by the same user for the same hotel are aggregated BEFORE
# any train/validation/test split.
#
# Therefore the same user-hotel pair cannot leak across splits.
# ==============================================================================

print(
    "\n[4] Deduplicating / aggregating repeated user-hotel interactions..."
)

pair_columns = [
    'user_id',
    'hotel_id',
    'img_hotel_id'
]


# Sanity check: img_hotel_id should map one-to-one to hotel_id.
mapping_check = (
    df_implicit[
        ['hotel_id', 'img_hotel_id']
    ]
    .drop_duplicates()
    .groupby('hotel_id')['img_hotel_id']
    .nunique()
)

bad_mapping = mapping_check[
    mapping_check > 1
]

if len(bad_mapping) > 0:

    raise ValueError(
        "Some hotel_id values map to multiple img_hotel_id values. "
        "Resolve this before creating v6."
    )


# --------------------------------------------------------------------------
# Preserve useful columns through aggregation
# --------------------------------------------------------------------------

if 'review_text' in df_implicit.columns:

    df_implicit['review_text'] = (
        df_implicit['review_text']
        .fillna('')
        .astype(str)
        .str.strip()
    )


def aggregate_pair(group):

    first = group.iloc[0].copy()

    # --------------------------------------------------------------
    # Combine every review belonging to this unique user-hotel pair.
    # --------------------------------------------------------------

    if 'review_text' in group.columns:

        texts = [
            t
            for t in group['review_text'].tolist()
            if t and t.lower() != 'nan'
        ]

        first['review_text'] = ' '.join(
            texts
        )


    # --------------------------------------------------------------
    # Since all rows already satisfy rating >= 4,
    # retain the maximum rating for the pair.
    # Rating is only relevant for constructing the positive set.
    # --------------------------------------------------------------

    if 'rating' in group.columns:

        first['rating'] = float(
            group['rating'].max()
        )


    first['label'] = 1

    return first


df_unique = (
    df_implicit
    .groupby(
        pair_columns,
        sort=False,
        group_keys=False
    )
    .apply(
        aggregate_pair
    )
    .reset_index(drop=True)
)


# --------------------------------------------------------------------------
# Duplicate audit BEFORE splitting
# --------------------------------------------------------------------------

duplicate_pairs = int(
    df_unique
    .duplicated(
        subset=[
            'user_id',
            'img_hotel_id'
        ]
    )
    .sum()
)

if duplicate_pairs != 0:

    raise RuntimeError(
        f"Deduplication failed: "
        f"{duplicate_pairs} duplicate user-hotel pairs remain."
    )


print(
    f"Rows before pair aggregation : "
    f"{len(df_implicit):,}"
)

print(
    f"Unique user-hotel interactions: "
    f"{len(df_unique):,}"
)

print(
    f"Removed repeated rows         : "
    f"{len(df_implicit) - len(df_unique):,}"
)

print(
    "✅ Exactly one row per user-hotel pair."
)


# ==============================================================================
# 7. ITEM-SIDE K-CORE
# ==============================================================================
#
# Apply the same >=5 positive interaction requirement, but now on UNIQUE
# user-hotel interactions rather than duplicated review rows.
# ==============================================================================

print(
    "\n[5] Applying item-side 5-core..."
)

item_counts = (
    df_unique[
        'img_hotel_id'
    ]
    .value_counts()
)

valid_items = set(
    item_counts[
        item_counts >= K_CORE_ITEM
    ]
    .index
)

df_unique = df_unique[
    df_unique['img_hotel_id']
    .isin(valid_items)
].copy()

print(
    f"Items with >=5 unique positive users : "
    f"{len(valid_items):,}"
)

print(
    f"Interactions after item k-core        : "
    f"{len(df_unique):,}"
)


# ==============================================================================
# 8. USER STRATIFICATION
# ==============================================================================

print(
    "\n[6] Building user groups..."
)

user_counts = (
    df_unique[
        'user_id'
    ]
    .value_counts()
)

dense_users = user_counts[
    user_counts >= DENSE_THRESHOLD
].index

sparse_users = user_counts[
    user_counts < DENSE_THRESHOLD
].index

df_dense = df_unique[
    df_unique['user_id']
    .isin(dense_users)
].copy()

df_sparse = df_unique[
    df_unique['user_id']
    .isin(sparse_users)
].copy()


print(
    f"Dense users (>=2 unique interactions): "
    f"{len(dense_users):,}"
)

print(
    f"Sparse users (<2)                 : "
    f"{len(sparse_users):,}"
)


# ==============================================================================
# 9. HYBRID 8:1:1 SPLIT
# ==============================================================================

np.random.seed(
    SEED
)


def stratified_split_v6(
    group
):

    indices = (
        group.index
        .to_numpy()
        .copy()
    )

    np.random.shuffle(
        indices
    )

    n = len(
        indices
    )


    # --------------------------------------------------------------
    # Exactly 2 interactions:
    # 1 train + 1 validation
    # No test.
    # --------------------------------------------------------------

    if n == 2:

        return pd.Series({
            'train': indices[:1],

            'val': indices[1:],

            'test': np.array(
                [],
                dtype=int
            )
        })


    # --------------------------------------------------------------
    # n >= 3:
    # approximately 80 / 10 / 10
    # --------------------------------------------------------------

    n_val = max(
        1,
        int(
            np.floor(
                n * 0.10
            )
        )
    )

    n_test = max(
        1,
        int(
            np.floor(
                n * 0.10
            )
        )
    )

    n_train = (
        n
        -
        n_val
        -
        n_test
    )


    # Safety
    if n_train < 1:

        n_train = 1

        if n_val > 1:
            n_val -= 1

        elif n_test > 1:
            n_test -= 1


    return pd.Series({

        'train': indices[
            :n_train
        ],

        'val': indices[
            n_train:
            n_train + n_val
        ],

        'test': indices[
            n_train + n_val:
        ]
    })


print(
    "\n[7] Performing user-stratified split..."
)

splits = (
    df_dense
    .groupby(
        'user_id',
        group_keys=False
    )
    .apply(
        stratified_split_v6
    )
)


# ==============================================================================
# 10. BUILD FINAL SPLITS
# ==============================================================================

train_indices = np.concatenate(
    [
        x
        for x in splits['train'].values
        if len(x) > 0
    ]
)

val_indices = np.concatenate(
    [
        x
        for x in splits['val'].values
        if len(x) > 0
    ]
)

test_indices = np.concatenate(
    [
        x
        for x in splits['test'].values
        if len(x) > 0
    ]
)


train_v6 = pd.concat(
    [
        df_dense.loc[
            train_indices
        ],

        df_sparse
    ],
    ignore_index=True
)

val_v6 = (
    df_dense
    .loc[
        val_indices
    ]
    .reset_index(
        drop=True
    )
)

test_v6 = (
    df_dense
    .loc[
        test_indices
    ]
    .reset_index(
        drop=True
    )
)

train_v6 = train_v6.reset_index(
    drop=True
)


# ==============================================================================
# 11. LEAKAGE AUDIT
# ==============================================================================

print(
    "\n[8] Running strict leakage audit..."
)


def make_pair_set(
    df
):

    return set(
        zip(
            df['user_id'].astype(str),
            df['img_hotel_id'].astype(str)
        )
    )


train_pairs = make_pair_set(
    train_v6
)

val_pairs = make_pair_set(
    val_v6
)

test_pairs = make_pair_set(
    test_v6
)


train_val_overlap = len(
    train_pairs & val_pairs
)

train_test_overlap = len(
    train_pairs & test_pairs
)

val_test_overlap = len(
    val_pairs & test_pairs
)


train_duplicates = int(
    train_v6
    .duplicated(
        subset=[
            'user_id',
            'img_hotel_id'
        ]
    )
    .sum()
)

val_duplicates = int(
    val_v6
    .duplicated(
        subset=[
            'user_id',
            'img_hotel_id'
        ]
    )
    .sum()
)

test_duplicates = int(
    test_v6
    .duplicated(
        subset=[
            'user_id',
            'img_hotel_id'
        ]
    )
    .sum()
)


print(
    f"Train duplicate pairs      : "
    f"{train_duplicates:,}"
)

print(
    f"Validation duplicate pairs : "
    f"{val_duplicates:,}"
)

print(
    f"Test duplicate pairs       : "
    f"{test_duplicates:,}"
)

print(
    f"Train ∩ Validation         : "
    f"{train_val_overlap:,}"
)

print(
    f"Train ∩ Test               : "
    f"{train_test_overlap:,}"
)

print(
    f"Validation ∩ Test          : "
    f"{val_test_overlap:,}"
)


assert train_duplicates == 0
assert val_duplicates == 0
assert test_duplicates == 0

assert train_val_overlap == 0
assert train_test_overlap == 0
assert val_test_overlap == 0


print(
    "✅ ZERO duplicate / cross-split user-hotel overlap."
)


# ==============================================================================
# 12. WARM-START CHECK
# ==============================================================================

train_users = set(
    train_v6[
        'user_id'
    ].astype(str)
)

train_items = set(
    train_v6[
        'img_hotel_id'
    ].astype(str)
)


for name, split_df in [
    ('validation', val_v6),
    ('test', test_v6)
]:

    split_users = set(
        split_df[
            'user_id'
        ].astype(str)
    )

    split_items = set(
        split_df[
            'img_hotel_id'
        ].astype(str)
    )

    cold_users = (
        split_users
        -
        train_users
    )

    cold_items = (
        split_items
        -
        train_items
    )

    print(
        f"{name.capitalize():<12} "
        f"cold users={len(cold_users):,}, "
        f"cold items={len(cold_items):,}"
    )

    assert len(cold_users) == 0
    assert len(cold_items) == 0


print(
    "✅ Warm-start verified."
)


# ==============================================================================
# 13. BUILD V6 CATALOG
# ==============================================================================

catalog_v6 = pd.DataFrame(
    {
        'img_hotel_id': sorted(
            valid_items
        )
    }
)


# ==============================================================================
# 14. USER / ITEM MAPS
# ==============================================================================

unique_users = sorted(
    set(
        train_v6['user_id']
    )
    |
    set(
        val_v6['user_id']
    )
    |
    set(
        test_v6['user_id']
    )
)

unique_items = sorted(
    catalog_v6[
        'img_hotel_id'
    ].astype(str)
    .unique()
)


user2idx = {
    str(uid): idx
    for idx, uid in enumerate(
        unique_users
    )
}

item2idx = {
    str(iid): idx
    for idx, iid in enumerate(
        unique_items
    )
}


with open(
    os.path.join(
        DRIVE_PATH,
        'user_map_v6.json'
    ),
    'w'
) as f:

    json.dump(
        user2idx,
        f
    )


with open(
    os.path.join(
        DRIVE_PATH,
        'item_map_v6.json'
    ),
    'w'
) as f:

    json.dump(
        item2idx,
        f
    )


# ==============================================================================
# 15. TRAIN HISTORY
# ==============================================================================

hist_v6 = (
    train_v6
    .groupby(
        'user_id'
    )['img_hotel_id']
    .apply(
        lambda s: sorted(
            set(
                s.astype(str)
            )
        )
    )
    .reset_index(
        name='train_items'
    )
)


# ==============================================================================
# 16. LEAK-FREE NEGATIVE SAMPLING
# ==============================================================================
#
# Negatives are sampled only from hotels the user has NEVER interacted with
# anywhere in the full deduplicated positive interaction set.
#
# Evaluation itself still uses full-catalog ranking rather than these negatives.
# ==============================================================================

print(
    "\n[9] Building 4:1 training negatives..."
)

catalog_list = (
    catalog_v6[
        'img_hotel_id'
    ]
    .astype(str)
    .to_numpy()
)


all_positive_pairs = pd.concat(
    [
        train_v6[
            ['user_id', 'img_hotel_id']
        ],

        val_v6[
            ['user_id', 'img_hotel_id']
        ],

        test_v6[
            ['user_id', 'img_hotel_id']
        ]
    ],
    ignore_index=True
)


user_all_seen_map = (
    all_positive_pairs
    .groupby(
        'user_id'
    )['img_hotel_id']
    .apply(
        lambda s: set(
            s.astype(str)
        )
    )
    .to_dict()
)


negative_rows = []


for user_id, pos_items in (
    train_v6
    .groupby(
        'user_id'
    )['img_hotel_id']
):

    user_id = str(
        user_id
    )

    positive_count = len(
        pos_items
    )

    n_negatives = (
        positive_count * 4
    )

    seen = user_all_seen_map[
        user_id
    ]

    collected = []

    while len(collected) < n_negatives:

        candidates = np.random.choice(
            catalog_list,
            size=max(
                32,
                n_negatives * 2
            ),
            replace=True
        )

        for candidate in candidates:

            candidate = str(
                candidate
            )

            if candidate in seen:
                continue

            collected.append(
                candidate
            )

            if len(collected) >= n_negatives:
                break


    for negative_item in collected:

        negative_rows.append(
            {
                'user_id': user_id,
                'img_hotel_id': negative_item,
                'label': 0
            }
        )


positive_rows = (
    train_v6[
        ['user_id', 'img_hotel_id']
    ]
    .copy()
)

positive_rows['label'] = 1


train_aug_v6 = pd.concat(
    [
        positive_rows,

        pd.DataFrame(
            negative_rows
        )
    ],
    ignore_index=True
)


# ==============================================================================
# 17. SAVE V6 FILES
# ==============================================================================

print(
    "\n[10] Saving v6 artifacts..."
)


train_v6.to_parquet(
    os.path.join(
        DRIVE_PATH,
        'train_v6.parquet'
    ),
    index=False
)

val_v6.to_parquet(
    os.path.join(
        DRIVE_PATH,
        'val_v6.parquet'
    ),
    index=False
)

test_v6.to_parquet(
    os.path.join(
        DRIVE_PATH,
        'test_v6.parquet'
    ),
    index=False
)

train_aug_v6.to_parquet(
    os.path.join(
        DRIVE_PATH,
        'train_augmented_v6.parquet'
    ),
    index=False
)

catalog_v6.to_parquet(
    os.path.join(
        DRIVE_PATH,
        'evaluation_catalog_items_v6.parquet'
    ),
    index=False
)

hist_v6.to_parquet(
    os.path.join(
        DRIVE_PATH,
        'train_history_v6.parquet'
    ),
    index=False
)


# ==============================================================================
# 18. FINAL V6 SUMMARY
# ==============================================================================

print("\n")
print("=" * 85)
print("FINAL V6 SPLIT SUMMARY")
print("=" * 85)

print(
    f"Unique users mapped       : "
    f"{len(user2idx):,}"
)

print(
    f"Catalog hotels            : "
    f"{len(item2idx):,}"
)

print(
    f"Train interactions        : "
    f"{len(train_v6):,}"
)

print(
    f"Validation interactions   : "
    f"{len(val_v6):,}"
)

print(
    f"Test interactions         : "
    f"{len(test_v6):,}"
)

print(
    f"Train users               : "
    f"{train_v6['user_id'].nunique():,}"
)

print(
    f"Validation users          : "
    f"{val_v6['user_id'].nunique():,}"
)

print(
    f"Test users                : "
    f"{test_v6['user_id'].nunique():,}"
)

print(
    f"Training negatives        : "
    f"{len(train_aug_v6) - len(train_v6):,}"
)

print(
    f"Train/Val overlap         : "
    f"{train_val_overlap:,}"
)

print(
    f"Train/Test overlap        : "
    f"{train_test_overlap:,}"
)

print(
    f"Val/Test overlap          : "
    f"{val_test_overlap:,}"
)

print("=" * 85)

print(
    "✅ V6 COMPLETE — CLEAN USER-HOTEL SPLIT"
)

print(
    "✅ One unique user-hotel interaction before splitting"
)

print(
    "✅ Zero duplicate pairs"
)

print(
    "✅ Zero cross-split user-hotel overlap"
)

print(
    "✅ Warm-start verified"
)

print(
    "✅ V6 maps/catalog/history/negatives saved"
)

print("=" * 85)

# %% PUBLIC NOTEBOOK CELL 11
# %% PUBLIC NOTEBOOK CELL 12
# ==============================================================================
# CLIP ViT-B/32 PER-IMAGE FEATURE EXTRACTION FOR THE FINAL V6 CATALOG
# ==============================================================================
import os
import gc
import threading
from io import BytesIO
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import CLIPModel

# ------------------------------------------------------------------ CONFIG ---
DRIVE_PATH = os.environ.get(
    'SAVREC_DATA_ROOT',
    '/content/drive/MyDrive/MS_new/Thesis_HotelRec_Data'
)

REGISTRY_FILE = os.path.join(DRIVE_PATH, 'item_images_v3.parquet')
CATALOG_FILE = os.path.join(DRIVE_PATH, 'evaluation_catalog_items_v6.parquet')

CLIP_PER_IMAGE_FILE = os.path.join(DRIVE_PATH, 'clip_per_image_v4.pt')
CLIP_COVERAGE_FILE = os.path.join(DRIVE_PATH, 'clip_coverage_v4.csv')
CLIP_FAILURE_FILE = os.path.join(DRIVE_PATH, 'clip_failures_v4.csv')

MAX_WORKERS = 16
GPU_BATCH = 64
DOWNLOAD_SLICE = 256
TIMEOUT = 8
MIN_PIXELS = 32

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# ------------------------------------------------------------- FINAL CATALOG ---
registry_df = pd.read_parquet(REGISTRY_FILE).copy()
registry_df['img_hotel_id'] = registry_df['img_hotel_id'].astype(str)

catalog_df = pd.read_parquet(CATALOG_FILE).copy()
catalog_df['img_hotel_id'] = catalog_df['img_hotel_id'].astype(str)
catalog_ids = set(catalog_df['img_hotel_id'])

registry_df = (
    registry_df[registry_df['img_hotel_id'].isin(catalog_ids)]
    .sort_values('img_hotel_id')
    .reset_index(drop=True)
)

if set(registry_df['img_hotel_id']) != catalog_ids:
    missing_registry = sorted(catalog_ids - set(registry_df['img_hotel_id']))
    raise RuntimeError(
        f'{len(missing_registry)} final V6 catalog hotels are missing from '
        f'{os.path.basename(REGISTRY_FILE)}; first examples: {missing_registry[:10]}'
    )

print(
    f'Final V6 catalog hotels: {len(catalog_ids):,} | '
    f'candidate image URLs: {int(registry_df["image_count"].sum()):,}'
)

# --------------------------------------------------------------- CLIP MODEL ---
print('Loading CLIP ViT-B/32...')
clip_model = (
    CLIPModel
    .from_pretrained('openai/clip-vit-base-patch32')
    .float()
    .to(device)
    .eval()
)

# This is the same image preprocessing used in the paper pipeline.
clip_tf = transforms.Compose([
    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])

# ------------------------------------------------------------- DOWNLOADER ---
_local = threading.local()

def _session():
    if not hasattr(_local, 'session'):
        s = requests.Session()
        s.headers.update({'User-Agent': 'Mozilla/5.0 (academic research)'})
        _local.session = s
    return _local.session

def _fetch(task):
    hotel_id, url = task
    try:
        response = _session().get(url, timeout=TIMEOUT)
        if response.status_code != 200:
            return hotel_id, None, f'http_{response.status_code}'
        if not response.headers.get('Content-Type', '').lower().startswith('image/'):
            return hotel_id, None, 'not_image'

        image = Image.open(BytesIO(response.content)).convert('RGB')
        if min(image.size) < MIN_PIXELS:
            return hotel_id, None, 'too_small'

        return hotel_id, clip_tf(image), 'ok'
    except Exception as exc:
        return hotel_id, None, type(exc).__name__

@torch.no_grad()
def _encode(batch_tensors):
    pixel_values = torch.stack(batch_tensors).to(device)
    output = clip_model.get_image_features(pixel_values=pixel_values)

    # transformers versions differ in the return wrapper.
    if hasattr(output, 'image_embeds'):
        output = output.image_embeds
    elif hasattr(output, 'pooler_output'):
        output = output.pooler_output

    output = output / output.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return output.detach().cpu().numpy().astype(np.float16)

# --------------------------------------------------------------- EXTRACTION ---
clip_per_image = {}
coverage_rows = []
failure_counts = defaultdict(int)

for _, row in tqdm(
    registry_df.iterrows(),
    total=len(registry_df),
    desc='hotels'
):
    hotel_id = str(row['img_hotel_id'])
    urls = row['available_images']
    if urls is None:
        urls = []
    urls = list(urls)

    hotel_vectors = []
    tensor_buffer = []

    def flush():
        if not tensor_buffer:
            return
        encoded = _encode(tensor_buffer)
        hotel_vectors.extend([v for v in encoded])
        tensor_buffer.clear()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for start in range(0, len(urls), DOWNLOAD_SLICE):
            tasks = [(hotel_id, u) for u in urls[start:start + DOWNLOAD_SLICE]]
            for _, image_tensor, reason in pool.map(_fetch, tasks):
                if image_tensor is None:
                    failure_counts[reason] += 1
                    continue

                tensor_buffer.append(image_tensor)
                if len(tensor_buffer) >= GPU_BATCH:
                    flush()
            flush()

    if hotel_vectors:
        arr = np.stack(hotel_vectors).astype(np.float16)
        clip_per_image[hotel_id] = arr
        n_embedded = int(arr.shape[0])
    else:
        n_embedded = 0

    coverage_rows.append({
        'img_hotel_id': hotel_id,
        'n_urls': int(len(urls)),
        'n_embedded': n_embedded,
        'has_visual': int(n_embedded > 0),
    })

coverage_df = pd.DataFrame(coverage_rows)
coverage_df.to_csv(CLIP_COVERAGE_FILE, index=False)

if failure_counts:
    pd.DataFrame(
        [{'reason': k, 'count': int(v)} for k, v in sorted(failure_counts.items())]
    ).to_csv(CLIP_FAILURE_FILE, index=False)

# The reported V7 benchmark contains at least one valid image for every final item.
missing_clip = sorted(catalog_ids - set(clip_per_image))
if missing_clip:
    raise RuntimeError(
        f'Fresh download produced no valid CLIP image for {len(missing_clip)} '
        f'final catalog hotels. External image availability has drifted. '
        f'Use the frozen clip_per_image_v4.pt for exact paper reproduction. '
        f'First missing IDs: {missing_clip[:10]}'
    )

torch.save(clip_per_image, CLIP_PER_IMAGE_FILE)

print('=' * 80)
print('CLIP EXTRACTION COMPLETE')
print('=' * 80)
print(f'Hotels encoded : {len(clip_per_image):,}')
print(f'Output          : {CLIP_PER_IMAGE_FILE}')
print(f'Coverage report : {CLIP_COVERAGE_FILE}')
print(
    'NOTE: exact paper reproduction should use the frozen artifact if '
    'external image URLs have changed.'
)

del clip_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# %% PUBLIC NOTEBOOK CELL 13
# ==============================================================================
# CELL 8 (v6): LOAD V6 DATA + VERIFY VISUAL FEATURE COVERAGE
# ==============================================================================
#
# Purpose:
#   Load the clean V6 train/validation/test split and verify that the existing
#   cached CLIP image embeddings contain all V6 catalog hotels.
#
# This cell does NOT train anything.
# ==============================================================================

import os
import json
import torch
import numpy as np
import pandas as pd


# ==============================================================================
# 1. CONFIG
# ==============================================================================

DRIVE_PATH = os.environ.get(
    'SAVREC_DATA_ROOT',
    '/content/drive/MyDrive/MS_new/Thesis_HotelRec_Data'
)

TRAIN_FILE = os.path.join(
    DRIVE_PATH,
    'train_v6.parquet'
)

VAL_FILE = os.path.join(
    DRIVE_PATH,
    'val_v6.parquet'
)

TEST_FILE = os.path.join(
    DRIVE_PATH,
    'test_v6.parquet'
)

CATALOG_FILE = os.path.join(
    DRIVE_PATH,
    'evaluation_catalog_items_v6.parquet'
)

HISTORY_FILE = os.path.join(
    DRIVE_PATH,
    'train_history_v6.parquet'
)

USER_MAP_FILE = os.path.join(
    DRIVE_PATH,
    'user_map_v6.json'
)

ITEM_MAP_FILE = os.path.join(
    DRIVE_PATH,
    'item_map_v6.json'
)

CLIP_PER_IMAGE_FILE = os.path.join(
    DRIVE_PATH,
    'clip_per_image_v4.pt'
)

print("=" * 85)
print("CELL 8 (v6): LOAD V6 DATA + VERIFY VISUAL FEATURE COVERAGE")
print("=" * 85)


# ==============================================================================
# 2. CHECK REQUIRED FILES
# ==============================================================================

required_files = [
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    CATALOG_FILE,
    HISTORY_FILE,
    USER_MAP_FILE,
    ITEM_MAP_FILE,
    CLIP_PER_IMAGE_FILE
]

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file is missing:\n{path}"
        )

print("✅ All required V6/data files found.")


# ==============================================================================
# 3. LOAD V6 DATA
# ==============================================================================

train_df = pd.read_parquet(
    TRAIN_FILE
)

val_df = pd.read_parquet(
    VAL_FILE
)

test_df = pd.read_parquet(
    TEST_FILE
)

catalog_df = pd.read_parquet(
    CATALOG_FILE
)

train_history_df = pd.read_parquet(
    HISTORY_FILE
)


# Normalize identifiers
for df_ in [
    train_df,
    val_df,
    test_df,
    catalog_df
]:

    if 'user_id' in df_.columns:

        df_['user_id'] = (
            df_['user_id']
            .astype(str)
            .str.strip()
        )

    if 'hotel_id' in df_.columns:

        df_['hotel_id'] = (
            df_['hotel_id']
            .astype(str)
            .str.strip()
        )

    if 'img_hotel_id' in df_.columns:

        df_['img_hotel_id'] = (
            df_['img_hotel_id']
            .astype(str)
            .str.strip()
        )


# ==============================================================================
# 4. LOAD MAPS
# ==============================================================================

with open(
    USER_MAP_FILE,
    'r'
) as f:

    user2idx = json.load(f)


with open(
    ITEM_MAP_FILE,
    'r'
) as f:

    item2idx = json.load(f)


# Ensure JSON keys are strings and indices are integers
user2idx = {
    str(k): int(v)
    for k, v in user2idx.items()
}

item2idx = {
    str(k): int(v)
    for k, v in item2idx.items()
}


num_users = len(
    user2idx
)

num_items = len(
    item2idx
)


# ==============================================================================
# 5. BUILD TRAINING HISTORY MASK
# ==============================================================================

train_user_mask_dict = {}

for user_id, group in (
    train_df
    .groupby('user_id')
):

    if user_id not in user2idx:
        continue

    user_idx = user2idx[
        user_id
    ]

    seen = [
        item2idx[str(item)]
        for item in group['img_hotel_id']
        if str(item) in item2idx
    ]

    train_user_mask_dict[
        user_idx
    ] = sorted(
        set(seen)
    )


# ==============================================================================
# 6. LOAD CACHED PER-IMAGE CLIP EMBEDDINGS
# ==============================================================================

print(
    "\nLoading cached per-image CLIP embeddings..."
)

per_img = torch.load(
    CLIP_PER_IMAGE_FILE,
    weights_only=False
)

per_img_ids = set(
    str(x)
    for x in per_img.keys()
)

v6_catalog_ids = set(
    catalog_df[
        'img_hotel_id'
    ]
    .astype(str)
)

missing_visual = (
    v6_catalog_ids
    -
    per_img_ids
)

extra_visual = (
    per_img_ids
    -
    v6_catalog_ids
)


print(
    f"V6 catalog hotels       : "
    f"{len(v6_catalog_ids):,}"
)

print(
    f"CLIP image hotels       : "
    f"{len(per_img_ids):,}"
)

print(
    f"Missing V6 visual hotels: "
    f"{len(missing_visual):,}"
)

print(
    f"Extra cached hotels     : "
    f"{len(extra_visual):,}"
)


# ==============================================================================
# 7. HARD COVERAGE CHECK
# ==============================================================================

if len(missing_visual) > 0:

    print(
        "\n⚠️ WARNING:"
    )

    print(
        "Some V6 catalog hotels do not have cached CLIP "
        "image embeddings."
    )

    print(
        "First missing IDs:"
    )

    print(
        sorted(
            list(missing_visual)
        )[:20]
    )

    raise RuntimeError(
        "V6 visual feature coverage is incomplete. "
        "Do not continue to model training."
    )


print(
    "✅ Every V6 catalog hotel has cached CLIP image embeddings."
)


# ==============================================================================
# 8. IMAGE COUNTS
# ==============================================================================

image_counts = []

for hotel_id in sorted(
    v6_catalog_ids
):

    arr = np.asarray(
        per_img[
            hotel_id
        ],
        dtype=np.float32
    )

    image_counts.append(
        len(arr)
    )


image_counts = np.asarray(
    image_counts
)


print("\nV6 IMAGE COVERAGE")

print(
    f"Mean images/hotel   : "
    f"{image_counts.mean():.2f}"
)

print(
    f"Median images/hotel : "
    f"{np.median(image_counts):.0f}"
)

print(
    f"Minimum             : "
    f"{image_counts.min():.0f}"
)

print(
    f"Maximum             : "
    f"{image_counts.max():.0f}"
)


# ==============================================================================
# 9. DATASET SUMMARY
# ==============================================================================

print("\n")
print("=" * 85)
print("V6 DATASET SUMMARY")
print("=" * 85)

print(
    f"Users                 : "
    f"{num_users:,}"
)

print(
    f"Catalog hotels        : "
    f"{num_items:,}"
)

print(
    f"Train interactions    : "
    f"{len(train_df):,}"
)

print(
    f"Validation interactions: "
    f"{len(val_df):,}"
)

print(
    f"Test interactions     : "
    f"{len(test_df):,}"
)

print(
    f"Train users           : "
    f"{train_df['user_id'].nunique():,}"
)

print(
    f"Validation users      : "
    f"{val_df['user_id'].nunique():,}"
)

print(
    f"Test users            : "
    f"{test_df['user_id'].nunique():,}"
)

print("=" * 85)

print(
    "✅ CELL 8 (v6) COMPLETE"
)

print(
    "V6 data loaded and visual-feature coverage verified."
)

print("=" * 85)

# %% PUBLIC NOTEBOOK CELL 15
# ==============================================================================
# CELL 9 (v6): LEAK-FREE ITEM TEXT FEATURES
#                     SBERT (384-d) + CLIP-text (512-d) + TF-IDF/SVD (384-d)
#
# V6 FIX:
#   - Uses the clean v6 split.
#   - One unique (user, hotel) interaction exists before splitting.
#   - Only TRAIN-V6 interactions are used to construct hotel text features.
#   - Validation and test interactions are never used as training text.
#
# Outputs:
#   item_text_embeddings_sbert_v6.pt
#   item_text_embeddings_clip_v6.pt
#   item_text_embeddings_tfidf_v6.pt
#   text_coverage_v6.csv
# ==============================================================================

import os
import gc
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from transformers import CLIPModel, CLIPTokenizer

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "sentence-transformers"
        ],
        check=True
    )

    from sentence_transformers import SentenceTransformer


# ==============================================================================
# 1. CONFIG
# ==============================================================================

DRIVE_PATH = os.environ.get(
    'SAVREC_DATA_ROOT',
    '/content/drive/MyDrive/MS_new/Thesis_HotelRec_Data'
)

PARQUET_MASTER = os.path.join(
    DRIVE_PATH,
    'merged_hotelrec.parquet'
)

TRAIN_FILE = os.path.join(
    DRIVE_PATH,
    'train_v6.parquet'
)

VAL_FILE = os.path.join(
    DRIVE_PATH,
    'val_v6.parquet'
)

TEST_FILE = os.path.join(
    DRIVE_PATH,
    'test_v6.parquet'
)

CATALOG_FILE = os.path.join(
    DRIVE_PATH,
    'evaluation_catalog_items_v6.parquet'
)

SBERT_OUT = os.path.join(
    DRIVE_PATH,
    'item_text_embeddings_sbert_v6.pt'
)

CLIP_TEXT_OUT = os.path.join(
    DRIVE_PATH,
    'item_text_embeddings_clip_v6.pt'
)

TFIDF_OUT = os.path.join(
    DRIVE_PATH,
    'item_text_embeddings_tfidf_v6.pt'
)

TEXT_MANIFEST = os.path.join(
    DRIVE_PATH,
    'text_coverage_v6.csv'
)

CLIP_IMG_IN = os.path.join(
    DRIVE_PATH,
    'item_visual_embeddings_clip_v4.pt'
)

READ_BATCH = 500_000
SBERT_BATCH = 256
CLIP_BATCH = 256

SKIP_EXISTING = True

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


print("=" * 80)
print("CELL 9 (v6): LEAK-FREE ITEM TEXT FEATURES")
print("=" * 80)

print(
    f"Device: {DEVICE}"
)


# ==============================================================================
# 2. CHECK SOURCE FILES
# ==============================================================================

required_inputs = [
    PARQUET_MASTER,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    CATALOG_FILE
]

for path in required_inputs:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

print(
    "✅ All V6 text-feature input files found."
)


# ==============================================================================
# 3. SCHEMA CHECK
# ==============================================================================

print("\n[1] Checking master interaction schema...")

schema = pq.read_schema(
    PARQUET_MASTER
).names

print(
    f"    columns: {schema}"
)

if 'review_text' not in schema:

    raise ValueError(
        "'review_text' is missing from merged_hotelrec.parquet"
    )

print(
    "✅ review_text available."
)


# ==============================================================================
# 4. LOAD V6 SPLITS
# ==============================================================================

print(
    "\n[2] Loading V6 train / validation / test / catalog..."
)

train_df = pd.read_parquet(
    TRAIN_FILE
)

val_df = pd.read_parquet(
    VAL_FILE
)

test_df = pd.read_parquet(
    TEST_FILE
)

catalog_df = pd.read_parquet(
    CATALOG_FILE
)


# Normalize IDs
for d in [
    train_df,
    val_df,
    test_df
]:

    d['user_id'] = (
        d['user_id']
        .astype(str)
        .str.strip()
    )

    d['hotel_id'] = (
        d['hotel_id']
        .astype(str)
        .str.strip()
    )

    d['img_hotel_id'] = (
        d['img_hotel_id']
        .astype(str)
        .str.strip()
    )


catalog_df['img_hotel_id'] = (
    catalog_df['img_hotel_id']
    .astype(str)
    .str.strip()
)


catalog_items = set(
    catalog_df[
        'img_hotel_id'
    ]
    .unique()
)


print(
    f"    train rows   : {len(train_df):,}"
)

print(
    f"    validation   : {len(val_df):,}"
)

print(
    f"    test rows    : {len(test_df):,}"
)

print(
    f"    catalog      : {len(catalog_items):,}"
)


# ==============================================================================
# 5. V6 LEAKAGE CHECK
# ==============================================================================

print(
    "\n[3] Checking V6 user-hotel split integrity..."
)

train_pairs = set(
    zip(
        train_df['user_id'],
        train_df['hotel_id']
    )
)

val_pairs = set(
    zip(
        val_df['user_id'],
        val_df['hotel_id']
    )
)

test_pairs = set(
    zip(
        test_df['user_id'],
        test_df['hotel_id']
    )
)


train_val_overlap = (
    len(
        train_pairs
        &
        val_pairs
    )
)

train_test_overlap = (
    len(
        train_pairs
        &
        test_pairs
    )
)

val_test_overlap = (
    len(
        val_pairs
        &
        test_pairs
    )
)


train_duplicates = int(
    train_df
    .duplicated(
        subset=[
            'user_id',
            'hotel_id'
        ]
    )
    .sum()
)

val_duplicates = int(
    val_df
    .duplicated(
        subset=[
            'user_id',
            'hotel_id'
        ]
    )
    .sum()
)

test_duplicates = int(
    test_df
    .duplicated(
        subset=[
            'user_id',
            'hotel_id'
        ]
    )
    .sum()
)


print(
    f"    train duplicates      : "
    f"{train_duplicates:,}"
)

print(
    f"    validation duplicates : "
    f"{val_duplicates:,}"
)

print(
    f"    test duplicates       : "
    f"{test_duplicates:,}"
)

print(
    f"    train ∩ validation    : "
    f"{train_val_overlap:,}"
)

print(
    f"    train ∩ test          : "
    f"{train_test_overlap:,}"
)

print(
    f"    validation ∩ test     : "
    f"{val_test_overlap:,}"
)


assert train_duplicates == 0
assert val_duplicates == 0
assert test_duplicates == 0

assert train_val_overlap == 0
assert train_test_overlap == 0
assert val_test_overlap == 0

print(
    "✅ V6 split is clean."
)


# ==============================================================================
# 6. MAP HOTEL IDs
# ==============================================================================

print(
    "\n[4] Building hotel mapping..."
)

hotel2img = dict(
    zip(
        train_df['hotel_id'],
        train_df['img_hotel_id']
    )
)

print(
    f"    hotel mappings: "
    f"{len(hotel2img):,}"
)


# ==============================================================================
# 7. STREAM TRAINING-ONLY REVIEW TEXT
# ==============================================================================
#
# CRITICAL:
# Only reviews belonging to TRAIN-V6 user-hotel pairs are used.
#
# Since V6 contains one unique user-hotel interaction per pair, this is
# directly leak-free.
# ==============================================================================

print(
    "\n[5] Streaming TRAIN-V6 review text only..."
)

pf = pq.ParquetFile(
    PARQUET_MASTER
)

print(
    f"    row groups: "
    f"{pf.num_row_groups:,}"
)

print(
    f"    source rows: "
    f"{pf.metadata.num_rows:,}"
)


train_pair_set = train_pairs

parts = []


for batch in tqdm(
    pf.iter_batches(
        batch_size=READ_BATCH,
        columns=[
            'user_id',
            'hotel_id',
            'review_text'
        ]
    ),
    desc="    batches"
):

    ch = batch.to_pandas()

    ch['user_id'] = (
        ch['user_id']
        .astype(str)
        .str.strip()
    )

    ch['hotel_id'] = (
        ch['hotel_id']
        .astype(str)
        .str.strip()
    )

    # Keep ONLY training user-hotel pairs.
    mask = [
        pair in train_pair_set
        for pair in zip(
            ch['user_id'],
            ch['hotel_id']
        )
    ]

    m = ch[
        mask
    ]

    if len(m) == 0:
        continue

    m = m.copy()

    m['review_text'] = (
        m['review_text']
        .fillna('')
        .astype(str)
        .str.strip()
    )

    m = m[
        m['review_text'] != ''
    ]

    if len(m) == 0:
        continue

    m['img_hotel_id'] = (
        m['hotel_id']
        .map(hotel2img)
    )

    m = m.dropna(
        subset=[
            'img_hotel_id'
        ]
    )

    parts.append(
        m[
            [
                'user_id',
                'hotel_id',
                'img_hotel_id',
                'review_text'
            ]
        ]
    )


if not parts:

    raise RuntimeError(
        "No training-only review text could be recovered."
    )


reviews = pd.concat(
    parts,
    ignore_index=True
)

del parts
gc.collect()


# ==============================================================================
# 8. CHECK FOR UNEXPECTED DUPLICATES
# ==============================================================================

before = len(
    reviews
)

# There should normally be exactly one source row per V6 pair after the
# upstream aggregation. We keep a final safety check here.
reviews = (
    reviews
    .drop_duplicates(
        subset=[
            'user_id',
            'hotel_id'
        ],
        keep='first'
    )
)


after = len(
    reviews
)

print(
    f"\n    matched training-review rows : "
    f"{before:,}"
)

print(
    f"    after pair deduplication     : "
    f"{after:,}"
)

print(
    f"    V6 train pairs               : "
    f"{len(train_pairs):,}"
)


# ==============================================================================
# 9. HOTEL TEXT COVERAGE
# ==============================================================================

hotel_review_counts = (
    reviews
    .groupby(
        'img_hotel_id'
    )
    .size()
    .rename(
        'n_reviews'
    )
    .to_frame()
)


covered_text_items = set(
    hotel_review_counts.index.astype(str)
)

missing_text = (
    catalog_items
    -
    covered_text_items
)


print(
    f"\n    Catalog hotels              : "
    f"{len(catalog_items):,}"
)

print(
    f"    Hotels with train text      : "
    f"{len(covered_text_items):,}"
)

print(
    f"    Hotels missing train text   : "
    f"{len(missing_text):,}"
)


if missing_text:

    print(
        "    First missing hotels:"
    )

    print(
        sorted(
            list(missing_text)
        )[:10]
    )

    raise RuntimeError(
        "Some catalog hotels have no usable TRAIN-V6 review text."
    )


print(
    "✅ Every V6 catalog hotel has training-only text."
)


# ==============================================================================
# 10. SAVE TEXT COVERAGE MANIFEST
# ==============================================================================

hotel_review_counts.to_csv(
    TEXT_MANIFEST
)

print(
    f"    Coverage manifest saved: "
    f"{TEXT_MANIFEST}"
)


# ==============================================================================
# 11. REVIEW-LENGTH SUMMARY
# ==============================================================================

word_counts = (
    reviews[
        'review_text'
    ]
    .str.split()
    .str.len()
)

print(
    "\n    Review length statistics:"
)

print(
    f"      median : "
    f"{word_counts.median():.0f} words"
)

print(
    f"      mean   : "
    f"{word_counts.mean():.1f} words"
)

print(
    f"      p95    : "
    f"{word_counts.quantile(0.95):.0f} words"
)

print(
    f"      max    : "
    f"{int(word_counts.max()):,} words"
)


texts = (
    reviews[
        'review_text'
    ]
    .tolist()
)

hids = (
    reviews[
        'img_hotel_id'
    ]
    .astype(str)
    .to_numpy()
)


# ==============================================================================
# 12. HELPER: MEAN-POOL BY HOTEL
# ==============================================================================

def pool_by_hotel(
    vectors,
    hotel_ids
):

    vector_df = pd.DataFrame(
        vectors
    )

    vector_df['__hotel__'] = (
        hotel_ids
    )

    grouped = (
        vector_df
        .groupby(
            '__hotel__'
        )
        .mean()
    )

    return {
        str(hotel_id):
            row.to_numpy(
                dtype=np.float32
            )

        for hotel_id, row
        in grouped.iterrows()
    }


# ==============================================================================
# 13. SBERT — 384 DIMENSIONS
# ==============================================================================

print(
    "\n[6] Sentence-BERT "
    "all-MiniLM-L6-v2 (384-d)"
)

if (
    SKIP_EXISTING
    and
    os.path.exists(
        SBERT_OUT
    )
):

    sbert_dict = torch.load(
        SBERT_OUT,
        weights_only=False
    )

    print(
        f"    existing file loaded; "
        f"skipped encoding "
        f"({len(sbert_dict):,} hotels)"
    )

else:

    sbert = SentenceTransformer(
        'all-MiniLM-L6-v2',
        device=str(DEVICE)
    )

    sbert_raw = sbert.encode(
        texts,
        batch_size=SBERT_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    sbert_dict = pool_by_hotel(
        sbert_raw,
        hids
    )

    # Final dtype normalization
    sbert_dict = {
        str(k): np.asarray(
            v,
            dtype=np.float32
        )
        for k, v in sbert_dict.items()
    }

    torch.save(
        sbert_dict,
        SBERT_OUT
    )

    print(
        f"    saved {len(sbert_dict):,} hotels"
    )

    del sbert
    del sbert_raw

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==============================================================================
# 14. VERIFY SBERT COVERAGE
# ==============================================================================

missing_sbert = (
    catalog_items
    -
    set(
        map(
            str,
            sbert_dict.keys()
        )
    )
)

if missing_sbert:

    raise RuntimeError(
        f"SBERT missing {len(missing_sbert):,} catalog hotels."
    )

print(
    "    ✅ SBERT coverage complete."
)


# ==============================================================================
# 15. CLIP TEXT — 512 DIMENSIONS
# ==============================================================================
#
# This is kept because your existing multimodal diagnostic uses CLIP's shared
# text-image space.
# ==============================================================================

print(
    "\n[7] CLIP ViT-B/32 text encoder (512-d)"
)

if (
    SKIP_EXISTING
    and
    os.path.exists(
        CLIP_TEXT_OUT
    )
):

    clip_text_dict = torch.load(
        CLIP_TEXT_OUT,
        weights_only=False
    )

    print(
        f"    existing file loaded; "
        f"skipped encoding "
        f"({len(clip_text_dict):,} hotels)"
    )

else:

    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    clip_model = (
        CLIPModel
        .from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        .float()
        .to(DEVICE)
        .eval()
    )

    clip_chunks = []

    with torch.no_grad():

        for start in tqdm(
            range(
                0,
                len(texts),
                CLIP_BATCH
            ),
            desc="    CLIP batches"
        ):

            batch_texts = texts[
                start:
                start + CLIP_BATCH
            ]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            )

            encoded = {
                key: value.to(
                    DEVICE
                )
                for key, value in encoded.items()
            }

            text_output = (
                clip_model
                .text_model(
                    **encoded
                )
            )

            if hasattr(
                text_output,
                'pooler_output'
            ):

                pooled = (
                    text_output
                    .pooler_output
                )

            else:

                pooled = (
                    text_output[1]
                )

            projected = (
                clip_model
                .text_projection(
                    pooled
                )
            )

            projected = (
                projected
                /
                projected.norm(
                    dim=-1,
                    keepdim=True
                ).clamp(
                    min=1e-12
                )
            )

            clip_chunks.append(
                projected
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )


    clip_raw = np.vstack(
        clip_chunks
    )

    del clip_chunks

    gc.collect()

    clip_text_dict = pool_by_hotel(
        clip_raw,
        hids
    )

    clip_text_dict = {
        str(k): np.asarray(
            v,
            dtype=np.float32
        )
        for k, v in clip_text_dict.items()
    }

    torch.save(
        clip_text_dict,
        CLIP_TEXT_OUT
    )

    print(
        f"    saved {len(clip_text_dict):,} hotels"
    )

    del clip_model
    del tokenizer
    del clip_raw

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==============================================================================
# 16. VERIFY CLIP-TEXT COVERAGE
# ==============================================================================

missing_clip_text = (
    catalog_items
    -
    set(
        map(
            str,
            clip_text_dict.keys()
        )
    )
)

if missing_clip_text:

    raise RuntimeError(
        f"CLIP-text missing "
        f"{len(missing_clip_text):,} catalog hotels."
    )

print(
    "    ✅ CLIP-text coverage complete."
)


# ==============================================================================
# 17. TF-IDF + SVD — 384 DIMENSIONS
# ==============================================================================

print(
    "\n[8] TF-IDF → TruncatedSVD (384-d)"
)

if (
    SKIP_EXISTING
    and
    os.path.exists(
        TFIDF_OUT
    )
):

    tfidf_dict = torch.load(
        TFIDF_OUT,
        weights_only=False
    )

    print(
        f"    existing file loaded; "
        f"skipped fitting "
        f"({len(tfidf_dict):,} hotels)"
    )

else:

    # Combine TRAIN-V6 review text by hotel.
    hotel_texts = (
        reviews
        .groupby(
            'img_hotel_id'
        )['review_text']
        .apply(
            lambda s: ' '.join(
                s
            )
        )
        .to_dict()
    )


    catalog_sorted = sorted(
        catalog_items
    )

    corpus = [
        hotel_texts[h]
        for h in catalog_sorted
    ]


    vectorizer = TfidfVectorizer(
        max_features=20_000,
        stop_words='english'
    )

    X = vectorizer.fit_transform(
        corpus
    )


    max_svd_components = min(
        384,
        X.shape[0] - 1,
        X.shape[1] - 1
    )

    if max_svd_components < 1:

        raise RuntimeError(
            "Not enough vocabulary/rows for SVD."
        )


    svd = TruncatedSVD(
        n_components=max_svd_components,
        random_state=42
    )

    Z = svd.fit_transform(
        X
    )


    print(
        f"    vocabulary              : "
        f"{len(vectorizer.vocabulary_):,}"
    )

    print(
        f"    SVD dimensions          : "
        f"{max_svd_components}"
    )

    print(
        f"    explained variance     : "
        f"{svd.explained_variance_ratio_.sum():.3f}"
    )


    tfidf_dict = {

        hotel_id:
            Z[idx]
            .astype(
                np.float32
            )

        for idx, hotel_id
        in enumerate(
            catalog_sorted
        )
    }


    torch.save(
        tfidf_dict,
        TFIDF_OUT
    )

    print(
        f"    saved {len(tfidf_dict):,} hotels"
    )


# ==============================================================================
# 18. VERIFY TF-IDF COVERAGE
# ==============================================================================

missing_tfidf = (
    catalog_items
    -
    set(
        map(
            str,
            tfidf_dict.keys()
        )
    )
)

if missing_tfidf:

    raise RuntimeError(
        f"TF-IDF missing "
        f"{len(missing_tfidf):,} catalog hotels."
    )

print(
    "    ✅ TF-IDF/SVD coverage complete."
)


# ==============================================================================
# 19. OPTIONAL CLIP SHARED-SPACE SANITY CHECK
# ==============================================================================

print(
    "\n[9] CLIP image-text shared-space sanity check..."
)

if os.path.exists(
    CLIP_IMG_IN
):

    clip_img = torch.load(
        CLIP_IMG_IN,
        weights_only=False
    )


    shared_ids = sorted(
        catalog_items
        &
        set(
            map(
                str,
                clip_img.keys()
            )
        )
        &
        set(
            map(
                str,
                clip_text_dict.keys()
            )
        )
    )


    if len(shared_ids) >= 10:

        rng = np.random.RandomState(
            42
        )

        sample_size = min(
            500,
            len(shared_ids)
        )

        selected = [
            shared_ids[i]
            for i in rng.choice(
                len(shared_ids),
                sample_size,
                replace=False
            )
        ]


        image_matrix = np.stack(
            [
                np.asarray(
                    clip_img[h],
                    dtype=np.float32
                )
                for h in selected
            ]
        )


        text_matrix = np.stack(
            [
                np.asarray(
                    clip_text_dict[h],
                    dtype=np.float32
                )
                for h in selected
            ]
        )


        image_matrix /= (
            np.linalg.norm(
                image_matrix,
                axis=1,
                keepdims=True
            )
            .clip(
                min=1e-12
            )
        )

        text_matrix /= (
            np.linalg.norm(
                text_matrix,
                axis=1,
                keepdims=True
            )
            .clip(
                min=1e-12
            )
        )


        matched_cos = float(
            (
                image_matrix
                *
                text_matrix
            )
            .sum(
                axis=1
            )
            .mean()
        )


        shuffled_indices = (
            rng.permutation(
                len(selected)
            )
        )


        random_cos = float(
            (
                image_matrix
                *
                text_matrix[
                    shuffled_indices
                ]
            )
            .sum(
                axis=1
            )
            .mean()
        )


        margin = (
            matched_cos
            -
            random_cos
        )


        print(
            f"    matched cosine : "
            f"{matched_cos:.4f}"
        )

        print(
            f"    shuffled cosine: "
            f"{random_cos:.4f}"
        )

        print(
            f"    margin         : "
            f"{margin:+.4f}"
        )


    else:

        print(
            "    skipped: insufficient shared hotels."
        )

else:

    print(
        f"    skipped: {CLIP_IMG_IN} not found."
    )


# ==============================================================================
# 20. FINAL SUMMARY
# ==============================================================================

coverage_stats = (
    hotel_review_counts[
        'n_reviews'
    ]
)

print("\n")
print("=" * 80)
print("V6 TEXT FEATURE EXTRACTION COMPLETE")
print("=" * 80)

print(
    f"Training interactions used : "
    f"{len(train_df):,}"
)

print(
    f"Unique training pairs used : "
    f"{len(train_pairs):,}"
)

print(
    f"Hotels with training text   : "
    f"{len(covered_text_items):,}"
)

print(
    f"SBERT        384-d          : "
    f"{len(sbert_dict):,} hotels"
)

print(
    f"CLIP-text    512-d          : "
    f"{len(clip_text_dict):,} hotels"
)

print(
    f"TF-IDF/SVD   {len(next(iter(tfidf_dict.values()))):3d}-d          : "
    f"{len(tfidf_dict):,} hotels"
)

print(
    f"Review count median         : "
    f"{coverage_stats.median():.0f}"
)

print(
    f"Review count mean           : "
    f"{coverage_stats.mean():.1f}"
)

print(
    f"Review count minimum        : "
    f"{coverage_stats.min():.0f}"
)

print(
    f"Review count maximum        : "
    f"{coverage_stats.max():.0f}"
)


print("\nSaved files:")

print(
    f"  {SBERT_OUT}"
)

print(
    f"  {CLIP_TEXT_OUT}"
)

print(
    f"  {TFIDF_OUT}"
)

print(
    f"  {TEXT_MANIFEST}"
)

print("=" * 80)
