# Klasifikacija metal podžanrova na osnovu audio karakteristika

Projekat iz predmeta **Softverski algoritmi u sistemima automatskog upravljanja**

**Student:** Marko Durić, RA 106-2023
**Profesor:** Darko Čapko
**Asistent:** Emina Demirović

## Opis projekta

Cilj projekta je razvoj modela koji na osnovu audio karakteristika pjesme (tempo, energy, danceability, itd.) predviđa kojem od četiri metal podžanra pjesma pripada:

- Black metal
- Death metal
- Symphonic metal
- Thrash metal

Skup podataka preuzet je sa Kaggle platforme i sadrži Spotify audio karakteristike pjesama.

## Struktura projekta

```
├── data/               # Sirovi i obrađeni podaci (CSV)
├── models/             # Sačuvani modeli, scaler i label encoder (.pkl)
├── notebooks/          # Jupyter notebook-ovi (EDA, feature engineering, modeling)
├── report/             # Word izvještaj sa rezultatima i zaključcima
├── results/            # Grafici i metrike izdvojeni iz notebook-ova
├── src/                # Pomoćne Python skripte za čišćenje sirovih podataka
├── app.py              # Streamlit aplikacija za predikciju
└── requirements.txt    # Lista potrebnih Python biblioteka
```

## Instalacija

1. Kloniraj repozitorijum:
```bash
git clone https://github.com/TvojeKorisnickoIme/metal-genre-classification.git
cd metal-genre-classification
```

2. Instaliraj potrebne biblioteke:
```bash
pip install -r requirements.txt
```

## Pokretanje notebook-ova

Notebook-ovi se pokreću ovim redoslijedom (svaki sledeći zavisi od fajlova koje prethodni sačuva u `data/processed/` i `models/`):

1. `notebooks/01_EDA.ipynb` — eksplorativna analiza podataka
2. `notebooks/02_feature_engineering.ipynb` — priprema podataka za modelovanje
3. `notebooks/03_modeling.ipynb` — treniranje, tuning i evaluacija modela

Otvori ih u Jupyteru, PyCharmu ili VS Code-u i pokreni **Restart & Run All**.

## Pokretanje Streamlit aplikacije

Nakon što su notebook-ovi pokrenuti i modeli sačuvani u `models/` folderu, pokreni:

```bash
streamlit run app.py
```

Aplikacija se otvara u browseru na adresi `http://localhost:8501`. Unesi audio karakteristike pjesme i klikni **"Predvidi podžanr"** da dobiješ predikciju sa pripadajućim vjerovatnoćama za sve četiri klase.

## Rezultati

| Model | Accuracy | F1-score |
|---|---|---|
| XGBoost (Tuned) | **83,2%** | **0,834** |
| XGBoost | 81,1% | 0,815 |
| Random Forest (Tuned) | 79,5% | 0,800 |
| Random Forest | 78,4% | 0,788 |
| SVM (RBF kernel) | 71,1% | 0,707 |
| KNN (K=1) | 63,2% | 0,631 |
| Logistička regresija | 58,4% | 0,581 |

Najbolji model je **XGBoost (Tuned)**, sačuvan u `models/xgboost_tuned.pkl`.

Detaljni rezultati, grafici i diskusija dostupni su u Word izvještaju u `report/` folderu.

## Korištene biblioteke

- pandas, numpy — obrada podataka
- matplotlib, seaborn — vizualizacija
- scikit-learn — modeli, preprocessing, metrike
- xgboost — gradient boosting model
- streamlit — web aplikacija za deployment
- joblib — čuvanje modela
