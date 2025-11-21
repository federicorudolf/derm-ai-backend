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
# META_CSV = "./Data/HIBA_Skin_Lesions.csv"
META_CSV = "./Data/metadata_limpia.csv"
# IMAGES_DIR = "./Data/HIBA_imagenes"
IMAGES_DIR = "./Data/Dataset-Clinico-Limpio"

COLS = {
  "image_id": "isic_id",
  "age_approx": "age_approx",
  "diagnosis_1": "diagnosis_1",
  "diagnosis_2": "diagnosis_2",
  "diagnosis_3": "diagnosis_3",
  "sex": "sex",
}
IMG_EXT = ".jpg"
DERM_SUBSTR = "derm"
MALIGNANT_LABELS = {
  "melanoma", "basal cell carcinoma", "bcc", "squamous cell carcinoma", "scc",
  "merkel cell carcinoma", "dermatofibrosarcoma", "intraepithelial carcinoma",
  "actinic keratosis", "lentigo maligna", "malignant" # Added "malignant"
}
BATCH_SIZE = 16  # Increased for better GPU utilization on M1
EPOCHS = 20
LR = 1e-4
WD = 1e-4
N_WORKERS = 0  # Set to 0 for M1 stability (avoids macOS semaphore leaks)

columns_to_drop = [
  'image_type',
  'acquisition_day',
  'copyright_license',
  'anatom_site_special',
  'diagnosis_4',
  'family_hx_mm',
  'fitzpatrick_skin_type',
  'mel_thick_mm',
  'melanocytic',
  'patient_id',
  'personal_hx_mm'
]

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
df = pd.read_csv(
    META_CSV,
    sep=";",           # <--- IMPORTANT: your file uses semicolons
    engine="c",        # fast C parser
    on_bad_lines="error"  # fail loudly if something is wrong
)
df.columns = [c.strip() for c in df.columns]  # remove any accidental spaces

print("Raw data loaded successfully")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
df = df.drop(columns=columns_to_drop, errors='ignore')

print("Columns dropped successfully. Shape:")
print(df.shape)

df_derm = df.copy()

df_derm["path"] = df_derm[COLS["image_id"]].astype(str).apply(lambda x: os.path.join(IMAGES_DIR, x + IMG_EXT))
df_derm = df_derm[df_derm["path"].apply(os.path.exists)].reset_index(drop=True)
print("Con archivo (after checking existence):", len(df_derm))

for key, col in COLS.items():
  if col not in df.columns:
    print(f"⚠️ COLS['{key}'] = '{col}' no coincide con ninguna columna en el CSV.")

benign_col = COLS.get("benign_malignant", None)

if COLS["diagnosis_1"] and COLS["diagnosis_1"] in df_derm.columns:
    df_derm["label"] = (
        df_derm[COLS["diagnosis_1"]]
        .astype(str)
        .str.lower()
        .apply(lambda s: int(any(lbl in s for lbl in MALIGNANT_LABELS)))
    )
elif benign_col and benign_col in df_derm.columns:
  df_derm["label"] = (
    df_derm[benign_col].astype(str).str.lower().str.contains("malig")
  ).astype(int)
else:
  raise ValueError("No encuentro columnas para crear la etiqueta.")

def is_malignant(s: str) -> int:
  tokens = {t.strip().lower() for t in s.replace(",", ";").split(";")}
  return int(any(t in MALIGNANT_LABELS for t in tokens))

df_derm["label"] = df_derm[COLS["diagnosis_1"]].astype(str).apply(is_malignant)

def stratified_split(
    df, test_size=0.10, val_size=0.10, group_col=None,
    seed=SEED, min_val_samples=10, max_retries=10
):
    y = df["label"].values
    initial_seed = seed

    for attempt in range(max_retries):
        current_seed = initial_seed + attempt

        if group_col is not None and group_col in df.columns and df[group_col].notna().any():
            # Proper stratified group split
            g = df[group_col].values
            print(f"Attempt {attempt+1}: Using StratifiedGroupKFold with group_col='{group_col}'")
            sss_test = StratifiedGroupKFold(n_splits=int(1/test_size), shuffle=True, random_state=current_seed)
            trainval_idx, test_idx = next(sss_test.split(X=np.zeros(len(y)), y=y, groups=g))
        else:
            print(f"Attempt {attempt+1}: Using StratifiedShuffleSplit (no valid group_col).")
            sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=current_seed)
            trainval_idx, test_idx = next(sss_test.split(np.zeros(len(y)), y))

        df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        sss_val = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_size / (1 - test_size),
            random_state=current_seed + 1
        )
        train_idx, val_idx = next(sss_val.split(np.zeros(len(df_trainval)), df_trainval["label"].values))
        df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
        df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

        if len(df_val) >= min_val_samples and df_val["label"].nunique() >= 2:
            print(f"Successfully created validation set after {attempt+1} attempts.")
            return df_train, df_val, df_test
        else:
            print(
                f"Attempt {attempt+1}: Validation set too small ({len(df_val)} samples) "
                f"or only {df_val['label'].nunique()} unique classes. Retrying..."
            )

    print("Max retries reached. Proceeding with the last validation set (AUC may be NaN).")
    return df_train, df_val, df_test


# Adjusting test_size and val_size to be smaller
df_train, df_val, df_test = stratified_split(df_derm, test_size=0.10, val_size=0.10, seed=SEED)
for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
  print(name, len(d), d["label"].mean())
  print(f"Label distribution in {name}:")
  print(d['label'].value_counts())

os.makedirs("./manifests/nov", exist_ok=True)
if not all(os.path.exists(f"./manifests/nov/derm_{split}.csv") for split in ["train", "val", "test"]):
  df_train.to_csv("./manifests/nov/derm_train.csv", index=False)
  df_val.to_csv("./manifests/nov/derm_val.csv", index=False)
  df_test.to_csv("./manifests/nov/derm_test.csv", index=False)
else:
  print("Manifests already exist")

IMG_SIZE = 448

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
    torch.save(model.state_dict(), "./checkpoints/derm_densenet121_best_nov.pth")
    print("  ↳ saved best model!")

print(f"\n✅ Training complete! Best validation AUC: {best_auc:.4f}")