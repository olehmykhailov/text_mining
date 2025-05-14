
# 🧠 Text Review Analyzer

This project is designed for **analyzing textual reviews** of Spotify`s app versions. It combines techniques like **clustering**, **topic modeling (LDA)**, **word association analysis**, **sentiment analysis**, and **visualization**, grouped by store or app version.

---

## 📊 Features

- ✅ Load and clean review data from a CSV file
- 🔍 Cluster reviews using TF-IDF and KMeans
- 📌 Extract and visualize dominant keywords per cluster
- 📚 Perform topic modeling using LDA
- 💬 Analyze sentiment of each review (e.g., using TextBlob or similar tools)
- 🧱 Generate association graphs (2-grams and 3-grams)
- 🕸 Build network graphs of frequent word associations
- 🔥 Compute word correlation heatmaps
- 📁 Automatically create separate folders for each app version
- ⚡ Parallel processing for performance boost

---

## 📂 Project Structure

```

project\_root/
│
├── main.py                         # Entry point for executing analysis
├── data/                           # Raw input CSV files
├── results/                         # Generated reports and visuals
├── scripts/
│   └── utilities.py                # Helper functions
|   └── clustering.py               # TF-IDF vectorization and clustering logic
|   └── topics_modeling.py          # LDA topic modeling
|   └── sentiment.py                # Sentiment analysis
|   └── text_associations.py        # N-gram & word association graphs
|   └── dimensionality_utils.py     # Dimensionality reduction & visualization
|   └── data_ops.py                 # Loading and pre-transforming data
├── requirements.txt                # Required Python packages
└── README.md                       # Project documentation

````

---

## 🚀 Getting Started

### 1. Install Dependencies

Make sure you have Python 3.7+ installed, then run:

```bash
pip install -r requirements.txt
````

### 2. Prepare Your Data

Your CSV should contain at least the following columns:

* `appVersion`
* `content`
* `score`


### 3. Run the Analysis

```bash
python main.py
```

Visuals and CSV outputs will be saved in the `results/` folder, organized by version.

---

## ⚙ Configuration

* Number of clusters, LDA topics, and other parameters can be changed in the respective function calls.
* Add filtering, preprocessing, or language support as needed.

---

## 🧼 Excluding Output from Git

To ignore generated folders:

```
# .gitignore
output/
__pycache__/
*.png
*.csv
```

---

## 📄 License

MIT License — feel free to use and modify.

---

## 🙌 Credits

Created using `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `TextBlob`, and more.
