# ₿ Bitcoin Price Direction Prediction using Machine Learning

A Machine Learning project that predicts whether Bitcoin's next closing price will increase using Logistic Regression, Decision Tree, and K-Nearest Neighbors (KNN), with data visualization, feature engineering, and model evaluation.


## Overview

This project analyzes historical Bitcoin price data and uses Machine Learning classification algorithms to predict the direction of the next closing price.

The project uses a dataset containing Bitcoin's opening, highest, lowest, and closing prices. Data visualization and feature engineering are performed to explore price movements and prepare the input features.

Three Machine Learning models are trained and evaluated:

* Logistic Regression
* Decision Tree
* K-Nearest Neighbors (KNN)

The models are evaluated using **ROC-AUC**, which measures their ability to distinguish between upward and non-upward price movements.


## Features

* Bitcoin historical price analysis
* Closing-price visualization using a logarithmic scale
* Price distribution visualization
* Yearly average price analysis
* Target distribution visualization
* Date-based feature extraction
* Feature engineering
* Min-Max-free standardization using `StandardScaler`
* Chronological train/validation split
* Logistic Regression classification
* Decision Tree classification
* K-Nearest Neighbors classification
* ROC-AUC evaluation
* Model performance comparison


## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn


## Dataset

The project uses the following dataset:

```text
bitcoin.csv
```

The dataset contains historical Bitcoin price information, including the following features:

| Feature | Description             |
| ------- | ----------------------- |
| `Date`  | Date of the observation |
| `Open`  | Opening price           |
| `High`  | Highest price           |
| `Low`   | Lowest price            |
| `Close` | Closing price           |

The project assumes the dataset is already ordered chronologically, from oldest to newest.

The target variable is:

```text
target
```

It represents the direction of the next closing price:

```text
1 → Next closing price is higher
0 → Next closing price is not higher
```

The goal is to predict price direction rather than the exact future price in dollars.


## Data Preprocessing

Several preprocessing steps are performed before training the models.

### Date Feature Extraction

The `Date` column is used to extract year, month, and day.

```python
splitted = df['Date'].str.split('-', expand=True)

df['year'] = splitted[0].astype('int')
df['month'] = splitted[1].astype('int')
df['day'] = splitted[2].astype('int')
```

These features allow the project to analyze price information over time.

### Feature Engineering

Two additional features are created using the daily prices:

```python
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
```

An additional feature identifies actual quarter-end dates:

```python
df['is_quarter_end'] = (
    pd.to_datetime(df['Date']).dt.is_quarter_end.astype(int)
)
```





## Author

**Mohammad Reza Bakhshandeh**

Electrical Engineering (Electronics) Graduate

Interested in Python Development, Machine Learning, Computer Vision, Artificial Intelligence, and Data Science.

