# Класификација метал поджанрова на основу аудио карактеристика

Пројекат из предмета **Софтверски алгоритми у системима аутоматског управљања**

**Студент:** Марко Дурић, РА 106-2023
**Професор:** Дарко Чапко
**Асистент:** Емина Демировић

## Опис пројекта

Циљ пројекта је развој модела који на основу аудио карактеристика пјесме (tempo, energy, danceability, итд.) предвиђа којем од четири метал поджанра пјесма припада:

- Black metal
- Death metal
- Symphonic metal
- Thrash metal

Скуп података преузет је са Kaggle платформе и садржи Spotify аудио карактеристике пјесама.

## Структура пројекта

```
├── data/               # Сирови и обрађени подаци (CSV)
├── models/             # Сачувани модели, scaler и label encoder (.pkl)
├── notebooks/          # Jupyter notebook-ови (EDA, feature engineering, modeling)
├── report/             # Word извјештај са резултатима и закључцима
├── results/            # Графици и метрике издвојени из notebook-ова
├── src/                # Помоћне Python скрипте за чишћење сирових података
├── app.py              # Streamlit апликација за предикцију
└── requirements.txt    # Листа потребних Python библиотека
```

## Инсталација

1. Клонирај репозиторијум:
```bash
git clone https://github.com/TvojeKorisnickoIme/metal-genre-classification.git
cd metal-genre-classification
```

2. Инсталирај потребне библиотеке:
```bash
pip install -r requirements.txt
```

## Покретање notebook-ова

Notebook-ови се покрећу овим редослиједом (сваки сљедећи зависи од фајлова које претходни сачува у `data/processed/` и `models/`):

1. `notebooks/01_EDA.ipynb` — експлоративна анализа података
2. `notebooks/02_feature_engineering.ipynb` — припрема података за моделовање
3. `notebooks/03_modeling.ipynb` — тренирање, tuning и евалуација модела

Отвори их у Jupyteru, PyCharmu или VS Code-у и покрени **Restart & Run All**.

## Покретање Streamlit апликације

Након што су notebook-ови покренути и модели сачувани у `models/` фолдеру, покрени:

```bash
streamlit run app.py
```

Апликација се отвара у browseru на адреси `http://localhost:8501`. Унеси аудио карактеристике пјесме и кликни **"Предвиди поджанр"** да добијеш предикцију са припадајућим вјероватноћама за све четири класе.

## Резултати

| Модел | Accuracy | F1-score |
|---|---|---|
| XGBoost (Tuned) | **83,2%** | **0,834** |
| XGBoost | 81,1% | 0,815 |
| Random Forest (Tuned) | 79,5% | 0,800 |
| Random Forest | 78,4% | 0,788 |
| SVM (RBF kernel) | 71,1% | 0,707 |
| KNN (K=1) | 63,2% | 0,631 |
| Логистичка регресија | 58,4% | 0,581 |

Најбољи модел је **XGBoost (Tuned)**, сачуван у `models/xgboost_tuned.pkl`.

Детаљни резултати, графици и дискусија доступни су у Word извјештају у `report/` фолдеру.

## Коришћене библиотеке

- pandas, numpy — обрада података
- matplotlib, seaborn — визуализација
- scikit-learn — модели, preprocessing, метрике
- xgboost — gradient boosting модел
- streamlit — web апликација за deployment
- joblib — чување модела
