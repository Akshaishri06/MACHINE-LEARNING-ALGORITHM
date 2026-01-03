import numpy as np
from sklearn.linear_model import LinearRegression
x=np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y=np.array([40,55,52,65,70,68,78,85,82,90])
model=LinearRegression()
model.fit(x,y)
m=model.coef_[0]
c=model.intercept_
print("slope(m):",round(m,2))
print("Intercept(x):",round(c,2))
x_new=np.array([[11]])
y_pred=model.predict(x_new)
print("predicted maarks for 11 hours:",round(y_pred[0],2))
