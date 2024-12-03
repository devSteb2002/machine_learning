import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('celsius-fahrenheit.csv', delimiter=',')
sns.scatterplot(x = "celsius", y = "fahrenheit", data = data,
                hue="fahrenheit", palette="coolwarm")

#caracteristicas (x), etiqueta (y)
x = data["celsius"]
y = data["fahrenheit"]

procesate_x = x.values.reshape(-1,1)
procesate_y = y.values.reshape(-1,1)

model = LinearRegression()

#enntrenamiento
model.fit(procesate_x, procesate_y)

#prediccion
#print(model.predict([[4567]]))
print(model.score(procesate_x, procesate_y))

# plt.show()