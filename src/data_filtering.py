from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

data_path = BASE_DIR / "data" / "processed" / "metal_4genre_dataset.csv"

df = pd.read_csv(data_path)
#df = pd.read_csv("metal_4genre_dataset.csv")

df.columns = df.columns.str.strip()

# osiguraj da nema NaN
df = df.dropna(subset=["genre"])

# helper funkcija
def sample_group(x, n=None):
    if n is None:
        return x
    return x.sample(n=min(len(x), n), random_state=42)

# prazna lista
frames = []

for genre, group in df.groupby("genre"):

    if genre in ["death-metal", "black-metal"]:
        frames.append(sample_group(group, 250))
    else:
        frames.append(sample_group(group))  # uzmi sve

balanced_df = pd.concat(frames).reset_index(drop=True)

# shuffle
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(balanced_df["genre"].value_counts())

balanced_df.to_csv("metal_mixed_sampling.csv", index=False)