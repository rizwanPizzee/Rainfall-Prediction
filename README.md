# Rainfall Prediction (Flask App)

## Purpose

This project is a small Flask web app that predicts whether it will rain today using a Random Forest model.
It provides a simple web form (index.html) for entering feature values and returns the model prediction.

---
## Webiste
 - Link : https://rainpredicttodayaus.pythonanywhere.com/
## View The Nootbook

- Link 1: https://github.com/rizwanPizzee/Rainfall-Prediction/blob/master/Data%20Analysis%20and%20ML.ipynb
- Link 2: https://nbviewer.org/github/rizwanPizzee/Rainfall-Prediction/blob/master/Data%20Analysis%20and%20ML.ipynb

## Dataset

- **Filename used in the notebook:** `weatherAUS.csv` (Link: https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package?resource=download&select=weatherAUS.csv)

---

## Tools / Tech stack

- **Language:** Python
- **Notebook environment:** Jupyter Notebook
- **Libraries used:**
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
- **Web Stack:**
  - `Flask`
  - `HTML`
  - `CSS`

---

## ML Models

- **Logistic Regression Model** — Supervised learning algorithm used to predict the rainfall.
    - Accuarcy: 83%
- **Random Forest Model** — Supervised learning algorithm used to predict the rainfall.
    - Accuracy: 84%   

---

## Project Structure

```
├── Data Analysis and ML.ipynb          # Nootbook conatins all EDA and ML related code
├── app.py                              # Flask web application
├── models.py                           # Machine learning pipeline
├── model.pkl                           # Trained model (generated)
├── weatherAUS.csv                      # dataset
├── templates/
│   ├── index.html                      # Main prediction interface
```
