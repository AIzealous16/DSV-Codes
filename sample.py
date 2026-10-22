import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {'Country': [1, 2, 3, 4, 5],
        'Number of Titles': [100, 200, 150, 300, 250]}

df = pd.DataFrame(data)
X = df[['Country']]
y = df['Number of Titles']

model = LinearRegression()
model.fit(X, y)
prediction = model.predict(X)

plt.scatter(X, y, color='blue')
plt.plot(X, prediction, color='red')
plt.xlabel("Country Code")
plt.ylabel("Number of Titles")
plt.title("Simple Linear Regression")
plt.show()

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)