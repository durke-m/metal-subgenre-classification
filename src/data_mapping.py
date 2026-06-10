from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

data_path = BASE_DIR / "data" / "raw" / "spotify-tracks-dataset-detailed.csv"

df = pd.read_csv(data_path)


# ---------------------------------------
# 1. Definiši mapping (PROŠIRI OVO kasnije)
# ---------------------------------------

genre_map = {
    # THRASH METAL
    "Slayer": "thrash-metal",
    "Megadeth": "thrash-metal",
    "Metallica": "thrash-metal",
    "Anthrax": "thrash-metal",
    "Kreator": "thrash-metal",
    "Sodom": "thrash-metal",
    "Testament": "thrash-metal",
    "Exodus": "thrash-metal",
    "Death Angel": "thrash-metal",
    "Sepultura": "thrash-metal",
    "Annihilator": "thrash-metal",
    "Suicidal Tendencies": "thrash-metal",
    "Municipal Waste": "thrash-metal",
    "Overkill": "thrash-metal",
    "Tankard": "thrash-metal",
    "Sabbat": "thrash-metal",
    

    # DEATH METAL
    #"Death": "death-metal",
    "Cannibal Corpse": "death-metal",
    "Morbid Angel": "death-metal",
    "Obituary": "death-metal",

    # BLACK METAL (UMJESTO DOOM)
    "Mayhem": "black-metal",
    "Burzum": "black-metal",
    "Darkthrone": "black-metal",
    "Emperor": "black-metal",
    "Immortal": "black-metal",
    "Gorgoroth": "black-metal",

    # SYMPHONIC METAL
    "Nightwish": "symphonic-metal",
    "Epica": "symphonic-metal",
    "Within Temptation": "symphonic-metal",
    "Therion": "symphonic-metal",
    "Delain": "symphonic-metal",
    "Visions of Atlantis": "symphonic-metal",
    "Xandria": "symphonic-metal",
    "Avantasia": "symphonic-metal",
    "Haggard": "symphonic-metal",
    "Beyond the black": "symphonic-metal",
    "Arkona": "symphonic-metal",
    "Korpiklaani": "symphonic-metal",
    "Alestorm": "symphonic-metal",


}

# ---------------------------------------
# 2. Kreiraj novu kolonu (copy original)
# ---------------------------------------

df["new_genre"] = df["track_genre"]

# ---------------------------------------
# 3. Override žanra po artistu
# ---------------------------------------

for artists, genre in genre_map.items():

    mask = df["artists"].str.contains(
        artists,
        case=False,
        na=False
    )

    df.loc[mask, "new_genre"] = genre

# ---------------------------------------
# 4. (opcionalno) izbaci ostale žanrove
# ---------------------------------------

df = df[
    df["new_genre"].isin([
        "thrash-metal",
        "death-metal",
        "black-metal",
        "symphonic-metal"
    ])
]

# ---------------------------------------
# 5. rename kolone i cleanup
# ---------------------------------------

df["track_genre"] = df["new_genre"]
df.drop(columns=["new_genre"], inplace=True)

# ---------------------------------------
# 6. save dataset
# ---------------------------------------

df.to_csv("metal_4genre_dataset.csv", index=False)

print(df["track_genre"].value_counts())
print(df.shape)