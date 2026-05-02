import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix, r2_score, root_mean_squared_error
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    ShuffleSplit,
    cross_val_score
)

'''
df_r = pd.read_csv('data/data_linear_regression.csv')

X = df_r.drop('price', axis=1)
y = df_r['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_dtr = DecisionTreeRegressor(max_depth=5, random_state=42)
model_dtr.fit(X_train, y_train)

y_pred = model_dtr.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
print(f'RMSE= {rmse:.2f}')

r2 = r2_score(y_test, y_pred)
print(f'R2= {r2:.2f}')

plot_tree(model_dtr, feature_names=X.columns, filled=True)
plt.title('Дерево решений (Регрессия)')
plt.savefig('tree_regression.png')
plt.show()
'''

df_c = pd.read_csv('data/data_logistic_regression.csv')

X = df_c.drop(columns=['loan_approved'], axis=1)
y = df_c['loan_approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_dtc = DecisionTreeClassifier(max_depth=5, random_state=42)
model_dtc.fit(X_train, y_train)

y_pred = model_dtc.predict(X_test)

print('Classification report: ')
print(classification_report(y_test, y_pred))

print('Confusion matrix: ')
print(' ', confusion_matrix(y_test, y_pred))

# plot_tree(model_dtc, feature_names=X.columns, class_names=['No Loan', 'Loan'], filled=True)
# plt.title('Дерево решений (Классификация)')
# plt.show()

# dtc_result = export_text(model_dtc, feature_names=X.columns, class_names=['No Loan', 'Loan'], filled=True)
# print(dtc_result)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_score = cross_val_score(model_dtc, X_train, y_train, cv=kfold, scoring='accuracy')

print('\n')
print(f'KFold Accuracy: {kfold_score.mean():.4f} (+/-) {kfold_score.std():.4f}')

stratfied_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
strat_scores = cross_val_score(model_dtc, X_train, y_train, cv=stratfied_kfold, scoring='accuracy')
print(f'Stratified KFold Accuracy: {strat_scores.mean():.4f} (+/-) {strat_scores.std():.4f}')

X_train_small = X_train.iloc[:100]
y_train_small = y_train.iloc[:100]

lou = LeaveOneOut()
lou_score = cross_val_score(model_dtc, X_train_small, y_train_small, cv=lou, scoring='accuracy')
print(f'LOU Accuracy: {lou_score.mean():.4f}')

shuffle_split = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
shuffle_scores = cross_val_score(model_dtc, X_train, y_train, cv=shuffle_split, scoring='accuracy')

print(f'Shuffle Split Accuracy: {shuffle_scores.mean():.4f} (+/-) {shuffle_scores.std():.4f}')