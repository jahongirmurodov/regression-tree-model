# Regression Tree Model for Predictive Analysis

## Overview
This project focuses on building a regression model using a Decision Tree algorithm to predict a continuous target variable. The goal was to explore how tree-based models can capture non-linear relationships in data and provide interpretable predictions.

## Problem Statement
Traditional linear models often fail to capture complex, non-linear relationships in data. The objective of this project was to apply a Decision Tree Regressor to improve prediction accuracy and better understand feature impact.

## Objectives
- Prepare and preprocess dataset  
- Train a Decision Tree regression model  
- Evaluate model performance  
- Analyze model behavior and predictions  

## Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib / Seaborn  

## Data Description
The dataset includes numerical and/or categorical features used to predict a continuous target variable.

Key steps included:
- Handling missing values (if present)  
- Splitting data into training and testing sets  

## Methodology

### 1. Data Preprocessing
- Loaded dataset and explored structure  
- Cleaned data and handled missing values  
- Encoded categorical variables (if applicable)  
- Split dataset into train/test sets  

### 2. Model Building
- Applied Decision Tree Regressor from Scikit-learn  
- Tuned key parameters such as:
  - max_depth  
  - min_samples_split  

### 3. Model Evaluation
- Evaluated model performance using metrics:
  - Mean Absolute Error (MAE)  
  - Mean Squared Error (MSE)  
  - R² Score  

### 4. Visualization & Interpretation
- Visualized decision tree structure  
- Analyzed how features influence predictions  
- Identified overfitting tendencies  

## Results
- Successfully trained a regression tree model  
- Captured non-linear relationships in the data  
- Observed that deeper trees may lead to overfitting  
- Achieved reasonable prediction accuracy (depending on dataset)

## Key Insights
- Decision Trees are powerful for non-linear problems  
- Model interpretability is a strong advantage  
- Proper tuning is required to avoid overfitting  

## Possible Improvements
- Apply ensemble methods (Random Forest, Gradient Boosting)  
- Perform hyperparameter tuning (GridSearchCV)  
- Use cross-validation  
- Compare with linear models


  # Регрессионное дерево для прогнозирования

## Обзор проекта
Проект посвящен построению модели регрессии с использованием алгоритма Decision Tree для предсказания числовой переменной. Основная цель — изучить, как деревья решений справляются с нелинейными зависимостями.

## Постановка задачи
Линейные модели часто не способны уловить сложные зависимости в данных. Задача проекта — применить Decision Tree Regressor для повышения точности прогнозирования и анализа влияния признаков.

## Цели проекта
- Подготовить и обработать данные  
- Обучить модель дерева решений  
- Оценить качество модели  
- Проанализировать поведение модели  

## Используемые технологии
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib / Seaborn  

## Описание данных
Датасет содержит числовые и/или категориальные признаки, используемые для предсказания целевой переменной.

Включает:
- Предобработку данных  
- Разделение на обучающую и тестовую выборки  

## Методология

### 1. Предобработка данных
- Загрузка и анализ структуры данных  
- Очистка данных и обработка пропусков  
- Кодирование категориальных признаков (при необходимости)  
- Разделение данных на train/test  

### 2. Построение модели
- Использование Decision Tree Regressor  
- Настройка параметров:
  - глубина дерева (max_depth)  
  - минимальное число объектов для разбиения  

### 3. Оценка модели
- Метрики качества:
  - MAE  
  - MSE  
  - R²  

### 4. Интерпретация
- Визуализация дерева решений  
- Анализ влияния признаков  
- Обнаружение переобучения  

## Результаты
- Построена модель регрессии  
- Выявлены нелинейные зависимости  
- Обнаружено переобучение при большой глубине дерева  
- Получена адекватная точность модели  

## Ключевые выводы
- Деревья решений эффективны для сложных зависимостей  
- Легко интерпретируются  
- Требуют настройки для избежания переобучения  

## Возможные улучшения
- Использование ансамблей (Random Forest, Gradient Boosting)  
- Подбор гиперпараметров  
- Кросс-валидация  
- Сравнение с другими моделями  
