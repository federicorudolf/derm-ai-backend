import os, random, json, math, time, shutil, glob, warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.amp as amp
import albumentations as A
import timm

from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix, f1_score
from albumentations.pytorch import ToTensorV2

# Enable MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

SEED = 128
IMG_SIZE = 448
META_CSV = "./Data/HIBA_Skin_Lesions.csv"
# META_CSV = "metadata_limpia.xlsx"
IMAGES_DIR = "./Data/HIBA_imagenes"
# IMAGES_DIR = "./Data/Dataset-Clinico-Limpio"

COLS = {
  "image_id": "isic_id",
  "filename": "",
  "image_type": "image_type",
  "diagnosis": "diagnosis_1",
  "benign_malignant": "benign_malignant",
  "patient_id": "patient_id",
  "lesion_id": "lesion_id"
}
IMG_EXT = ".jpg"
DERM_SUBSTR = "derm"
MALIGNANT_LABELS = {
  "melanoma", "basal cell carcinoma", "bcc", "squamous cell carcinoma", "scc",
  "merkel cell carcinoma", "dermatofibrosarcoma", "intraepithelial carcinoma",
  "actinic keratosis", "lentigo maligna", "malignant"
}
BATCH_SIZE = 16  # Increased for better GPU utilization on M1
EPOCHS = 20
LR = 1e-4
WD = 1e-4
N_WORKERS = 0  # Set to 0 for M1 stability (avoids macOS semaphore leaks)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore")

# Device setup
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    torch.mps.set_per_process_memory_fraction(0.0)  # Let MPS manage memory
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.cuda.manual_seed_all(SEED)
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")
print(f"Data loading workers: {N_WORKERS}")

# Read the HIBA dataset
df = pd.read_csv(META_CSV)
# df = pd.read_excel(META_CSV)

# --- Detectar columna 'image_type' ---
img_type_col = COLS["image_type"] if COLS["image_type"] in df.columns else None
if img_type_col is None:
    candidates = [c for c in df.columns if "type" in c.lower()]
    img_type_col = candidates[0] if candidates else None

# --- Filtrar dermatoscópicas ---
if img_type_col:
  derm_mask = df[img_type_col].astype(str).str.lower().str.contains(DERM_SUBSTR)
  df_derm = df[derm_mask].copy()
else:
  print("⚠️ No se detectó columna de tipo de imagen. Continuaremos con TODO el dataset (ajusta filtros).")
  df_derm = df.copy()

print("Cantidad de imágenes dermatoscópicas:", len(df_derm))

# --- Construir path de imagen ---
if COLS["filename"] and COLS["filename"] in df_derm.columns:
  df_derm["path"] = df_derm[COLS["filename"]].apply(lambda x: x if os.path.isabs(x) else os.path.join(IMAGES_DIR, str(x)))
elif COLS["image_id"] and COLS["image_id"] in df_derm.columns:
  df_derm["path"] = df_derm[COLS["image_id"]].astype(str).apply(lambda x: os.path.join(IMAGES_DIR, x + IMG_EXT))
else:
  raise ValueError("No encuentro cómo construir la ruta de imagen. Ajusta COLS['filename'] o COLS['image_id'].")

# --- Limpiar filas sin archivo ---
df_derm = df_derm[df_derm["path"].apply(os.path.exists)].reset_index(drop=True)
print("Cantidad de imágenes con archivo:", len(df_derm))

# Check minimum dataset size
if len(df_derm) < 50:
  print(f"⚠️  WARNING: Only {len(df_derm)} images found. Need at least 50 for proper train/val/test splits")
  print("   Consider using smaller split ratios or getting more data")

# --- Crear etiqueta binaria ---
if COLS["benign_malignant"] and COLS["benign_malignant"] in df_derm.columns:
  df_derm["label"] = (df_derm[COLS["benign_malignant"]].astype(str).str.lower().str.contains("malig")).astype(int)
elif COLS["diagnosis"] and COLS["diagnosis"] in df_derm.columns:
  df_derm["label"] = df_derm[COLS["diagnosis"]].astype(str).str.lower().apply(lambda s: int(any(x in s for x in MALIGNANT_LABELS)))
else:
  raise ValueError("No encuentro columnas para crear la etiqueta. Ajusta COLS['benign_malignant'] o COLS['diagnosis'].")

# --- ID para split por paciente/lesión ---
group_col = None
if COLS["patient_id"] and COLS["patient_id"] in df_derm.columns:
  group_col = COLS["patient_id"]
elif COLS["lesion_id"] and COLS["lesion_id"] in df_derm.columns:
  group_col = COLS["lesion_id"]

df_derm = df_derm[["path", "label"] + ([group_col] if group_col else [])].copy()
print(df_derm.head(3))
print(df_derm["label"].value_counts())

def stratified_split(df, test_size=0.15, val_size=0.15, group_col=None, seed=SEED):
  y = df["label"].values
  if group_col:
    groups = df[group_col].astype(str).values
    sss = StratifiedGroupKFold(n_splits=int(1/(test_size)), shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(sss.split(np.zeros(len(y)), y, groups))
    df_trainval, df_test = df.iloc[trainval_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    y_tv = df_trainval["label"].values
    groups_tv = df_trainval[group_col].astype(str).values
    sss2 = StratifiedGroupKFold(n_splits=int(1/val_size), shuffle=True, random_state=seed+1)
    train_idx, val_idx = next(sss2.split(np.zeros(len(y_tv)), y_tv, groups_tv))
    df_train, df_val = df_trainval.iloc[train_idx].reset_index(drop=True), df_trainval.iloc[val_idx].reset_index(drop=True)
  else:
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(sss_test.split(np.zeros(len(y)), y))
    df_trainval, df_test = df.iloc[trainval_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=seed+1)
    train_idx, val_idx = next(sss_val.split(np.zeros(len(df_trainval)), df_trainval["label"].values))
    df_train, df_val = df_trainval.iloc[train_idx].reset_index(drop=True), df_trainval.iloc[val_idx].reset_index(drop=True)
  return df_train, df_val, df_test


df_train, df_val, df_test = stratified_split(df_derm, test_size=0.15, val_size=0.15, group_col=group_col, seed=SEED)
print("\n📊 Dataset splits:")
for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
  n_positive = d["label"].sum()
  n_negative = len(d) - n_positive
  print(f"{name:>5}: {len(d):4d} images | {n_negative:4d} benign, {n_positive:4d} malignant | {d['label'].mean():.1%} positive")

# Check if we have both classes in validation
if df_val["label"].nunique() < 2:
  print("\n⚠️  WARNING: Validation set has only one class! Consider:")
  print("   - Using a different random seed")
  print("   - Adjusting split sizes")
  print("   - Checking if your dataset is very imbalanced")
  print("\n🔄 Retrying with different seed...")
  
  # Try with different seed
  df_train, df_val, df_test = stratified_split(df_derm, test_size=0.15, val_size=0.15, group_col=group_col, seed=SEED+42)
  
  print("\nNew splits:")
  for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
    n_positive = d["label"].sum()
    n_negative = len(d) - n_positive
    print(f"{name:>5}: {len(d):4d} images | {n_negative:4d} benign, {n_positive:4d} malignant | {d['label'].mean():.1%} positive")

# Guardar manifests
os.makedirs("./manifests", exist_ok=True)
if not all(os.path.exists(f"./manifests/derm_{split}.csv") for split in ["train", "val", "test"]):
  df_train.to_csv("./manifests/derm_train.csv", index=False)
  df_val.to_csv("./manifests/derm_val.csv", index=False)
  df_test.to_csv("./manifests/derm_test.csv", index=False)
else:
  print("Manifests already exist")

# Optimized transforms
train_tfms = A.Compose([
  A.LongestMaxSize(max_size=IMG_SIZE),
  A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_REFLECT_101),
  A.HorizontalFlip(p=0.5),
  A.VerticalFlip(p=0.2),
  A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=15, p=0.5),
  A.OneOf([A.RandomBrightnessContrast(0.1, 0.1), A.ColorJitter(0.05,0.05,0.05,0.05)], p=0.5),
  A.CLAHE(p=0.1),
  A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
  ToTensorV2()
])

val_tfms = A.Compose([
  A.LongestMaxSize(max_size=IMG_SIZE),
  A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_REFLECT_101),
  A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
  ToTensorV2()
])

class SkinDataset(Dataset):
  def __init__(self, df, augment=False):
    self.df = df.reset_index(drop=True)
    self.augment = augment
    self.tfm = train_tfms if augment else val_tfms
    # Preload images for faster training on small dataset
    self.images = []
    for path in self.df["path"]:
      try:
        img = Image.open(path).convert('RGB')
        img = np.array(img)
      except:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
          raise FileNotFoundError(f"Cannot load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      self.images.append(img)
    print(f"Preloaded {len(self.images)} images for {'train' if augment else 'val/test'}")
    
  def __len__(self): 
    return len(self.df)
    
  def __getitem__(self, idx):
    img = self.images[idx]
    label = torch.tensor(self.df.iloc[idx]["label"], dtype=torch.float32)
    img = self.tfm(image=img)["image"]
    return img, label

# Create model
model = timm.create_model('densenet121', pretrained=True, num_classes=1)
model.to(DEVICE)

# MPS uses float32
model = model.to(torch.float32)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Optimizer
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)

# DataLoaders
train_dataset = SkinDataset(df_train, augment=True)
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=N_WORKERS,
    pin_memory=False  # Not needed on M1 unified memory
)

val_dataset = SkinDataset(df_val, augment=False)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=N_WORKERS,
    pin_memory=False
)

# Loss
pos_weight = ( (len(train_dataset) - train_dataset.df['label'].sum()) / (train_dataset.df['label'].sum()+1e-6) )
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE))

def evaluate(model, loader, device=None, threshold=0.5):
  device = device or next(model.parameters()).device
  model.eval(); model.float()
  all_logits, all_y = [], []
  
  with torch.inference_mode():
    for x, y in loader:
      x = x.to(device); y = y.to(device)
      # Use autocast for evaluation as well
      with amp.autocast(device_type=device.type, enabled=(device.type != 'cpu')):
        logits = model(x).view(-1)
      all_logits.append(logits.detach().cpu())
      all_y.append(y.detach().cpu())
  all_logits = torch.cat(all_logits).numpy()
  all_y = torch.cat(all_y).numpy()
  probs = 1/(1+np.exp(-all_logits))
  preds = (probs >= threshold).astype(int)
  auc = roc_auc_score(all_y, probs)
  ap  = average_precision_score(all_y, probs)
  f1 = f1_score(all_y, preds)
  cm = confusion_matrix(all_y, preds)
  return auc, ap, f1, cm, probs, all_y


best_auc = 0.0
os.makedirs("./checkpoints", exist_ok=True)

# Training loop
print(f"\n🏃 Starting training... Like River Plate in the Monumental, let's dominate!\n")

for epoch in range(1, EPOCHS + 1):
  start_time = time.time()
  model.train()
  model.float()
  
  running = 0.0
  n_batches = 0
  
  for batch_idx, (x, y) in enumerate(train_loader):
    x = x.to(DEVICE, non_blocking=True)
    y = y.to(DEVICE, non_blocking=True)
    if not torch.isfinite(x).all():
      print(f"⚠️  Non-finite input at batch {batch_idx}, epoch {epoch}. Skipping...")
      continue
    optimizer.zero_grad(set_to_none=True)
    logits = model(x).view(-1)
    loss = criterion(logits, y)
    if not torch.isfinite(loss):
      print(f"⚠️  Non-finite loss at batch {batch_idx}, epoch {epoch}. Skipping...")
      continue
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    running += loss.item() * x.size(0)
    n_batches += 1
  if n_batches > 0:
    train_loss = running / len(train_dataset)
  else:
    print(f"⚠️  No valid batches in epoch {epoch}")
    continue
  auc, val_ap, f1, cm, probs, all_y = evaluate(model, val_loader)
  epoch_time = time.time() - start_time
  print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | val_auc {auc:.4f} | val_ap {val_ap:.4f} | val_f1 {f1:.4f} | time {epoch_time:.1f}s")
  print(f"  Confusion Matrix:")
  print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
  print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
  if auc > best_auc and auc > 0.5:
    best_auc = auc
    torch.save(model.state_dict(), "./checkpoints/derm_densenet121_best.pth")
    print("  ↳ saved best model!")

print(f"\n✅ Training complete! Best validation AUC: {best_auc:.4f}")