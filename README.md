<p align="center">
  <img src="./yelp-cover.png" width="600" alt="Yelp Project Banner"/>
</p>

<h1 align="center">🍽 Yelp Review Analysis: Dietary-Friendly Restaurant Insights</h1>

<p align="center">
  Identifying what matters most to customers with dietary restrictions through NLP and data analysis.
</p>

---

## 📊 Overview

This project analyzes Yelp restaurant reviews to uncover which features — such as **cuisine type, price level, location, and sentiment** — are most associated with **positive reviews from users with dietary needs** (e.g., vegan, gluten-free, halal).

The goal is to help:
- 🥗 **Consumers** find inclusive, well-reviewed restaurants
- 🍴 **Restaurants** improve offerings and marketing
- 📢 **Marketers** identify value signals in niche food segments
- 🔎 **Platforms** like Yelp enhance filter and recommendation systems

---

## 💡 Key Research Questions

- What business features correlate with 4–5 star reviews from users with dietary restrictions?
- How does review sentiment vary by cuisine, price level, or region?
- Can we predict high-rated reviews using structured metadata and unstructured text?

---

## 🧠 Methodology

1. **Data Sourcing**: Yelp Open Dataset (Restaurants & Reviews)
2. **Filtering**: Focused on U.S.-based restaurants with >100 reviews
3. **Keyword Extraction**: Identified reviews mentioning dietary terms
4. **Sentiment Analysis**: Using TextBlob and VADER to score polarity
5. **Feature Engineering**: Review length, price, cuisine, location, sentiment, etc.
6. **Modeling**: Trained classification models (Random Forest, XGBoost) to predict high satisfaction
7. **Insights**: Ranked feature importance and visualized patterns

---

## 🔧 Tech Stack

- **Languages**: Python
- **Libraries**:  
  - `pandas`, `numpy`, `scikit-learn`, `xgboost`  
  - `TextBlob`, `NLTK`, `VADER`, `spaCy`  
  - `matplotlib`, `seaborn`, `plotly`
- **Tools**: Jupyter Notebooks, Git, VSCode
- **Data**: Yelp Open Dataset (business.json, review.json)

---

## 📁 Folder Structure

yelp-review-analysis/
├── data/ # Cleaned & raw Yelp datasets
├── notebooks/ # Jupyter notebooks for EDA, modeling, evaluation
├── src/ # Python scripts for preprocessing and modeling
├── outputs/ # Graphs, summary reports, feature importances
├── yelp-cover.png # Banner image
├── README.md # This file
└── requirements.txt # Python dependencies


## 📈 Key Insights

- **Vegan and gluten-free restaurants** in urban, mid-to-high price ranges show higher sentiment scores and review ratings.
- **Informative and emotionally positive language** strongly correlates with 4–5 star reviews among dietary-concerned users.
- **Review length** and **sentiment polarity** are strong predictors of customer satisfaction.

---

## 🔮 Future Directions

- Incorporate **geospatial mapping** (e.g., `geopandas`) for regional insights
- Use **transformer-based sentiment models** (BERT or RoBERTa) for more accurate NLP
- Build a **simple web dashboard** to help users filter by dietary-friendly features

---

## 👩‍💻 Author

**Bonnie Yen**  
Master’s in Financial Analytics | Oregon State University  
📍 Salem, Oregon  
🔗 [LinkedIn](https://linkedin.com/in/bonnie-jing-jie-yen) | [GitHub](https://github.com/bonnb2010)

---
