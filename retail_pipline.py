import pandas as pd
import numpy as np 
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

np.random.seed(42)
dates = pd.date_range('2023-01-01', '2023-06-30')

data = {
    'transaction_id':range(1000),
    'customer_id':np.random.randint(100, 200, 1000),
    'product_category':np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], 1000),
    'purchase_amount':np.round(np.abs(np.random.normal(50,30,1000)),2),
    'time_spent':np.abs(np.random.normal(5,3,1000)),
    'date':np.random.choice(dates, 1000)
}
df = pd.DataFrame(data)


df['is_weekend'] = df['date'].dt.dayofweek >= 5
df['spending_tier'] = pd.cut(df['purchase_amount'],
                            bins=[0,25,75,np.inf],
                            labels=['Low','Medium','High'])



engine = create_engine('sqlite:///retail_sales.db')
df.to_sql('transactions', engine, if_exists='replace', index=False)


df_ml = pd.get_dummies(df, columns=['product_category', 'spending_tier'])
features = ['customer_id', 'time_spent', 'is_weekend', 
           'product_category_Books', 'product_category_Clothing',
           'product_category_Electronics', 'product_category_Home']

X = df_ml[features]
Y = df_ml['purchase_amount']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
model = RandomForestRegressor(n_estimators=50)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"MAE: ${mean_absolute_error(y_test, y_pred):.2f}")

plt.figure(figsize=(10,6))
plt.barh(features, model.feature_importances_)
plt.title('Feature Importance for Purchase Amount Prediction')
plt.savefig('feature_importance.png')