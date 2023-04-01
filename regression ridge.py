#!/usr/bin/env python
# coding: utf-8

# In[3]:



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error


# In[4]:


# Créer la dataset x et y
x = np.array([10.9, 12.4, 13.5, 14.6, 14.8, 15.6, 16.2, 17.5, 18.3, 18.6])
y = np.array([24.8, 30.0, 31.0, 29.3, 35.9, 36.9, 42.5, 37.9, 38.9, 40.5])


# In[5]:


# Définir les valeurs de alpha à utiliser pour Ridge
alphas = [0, 0.1, 1, 10, 100]


# In[6]:


# Entraîner un modèle de régression linéaire sur la dataset
lr = LinearRegression()
lr.fit(x.reshape(-1, 1), y)
lr_coef = lr.coef_[0]
lr_intercept = lr.intercept_


# In[7]:


# Calculer l'erreur quadratique moyenne (MSE) pour la régression linéaire
lr_mse = mean_squared_error(y, lr.predict(x.reshape(-1, 1)))


# In[8]:


# Tracer la courbe de la régression linéaire
plt.scatter(x, y, color='black')
plt.plot(x, lr_coef*x + lr_intercept, label='Régression linéaire')
plt.title('Régression linéaire sur la dataset')
plt.legend()
plt.show()


# In[9]:


# Entraîner un modèle de régression Ridge pour chaque valeur de alpha et stocker les coefficients
ridge_coefs = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(x.reshape(-1, 1), y)
    ridge_coefs.append(ridge.coef_[0])


# In[10]:


# Tracer la courbe de Ridge pour chaque valeur de alpha
plt.scatter(x, y, color='black')
for i, alpha in enumerate(alphas):
    plt.plot(x, ridge_coefs[i]*x, label='Ridge, alpha='+str(alpha))
plt.title('Régression Ridge sur la dataset')
plt.legend()
plt.show()


# In[11]:


# Tracer la courbe de l'erreur MSE en fonction de la valeur de alpha pour Ridge
ridge_mse_values = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(x.reshape(-1, 1), y)
    ridge_mse_values.append(mean_squared_error(y, ridge.predict(x.reshape(-1, 1))))

plt.plot(alphas, ridge_mse_values)
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.title("Ridge Regularization")
plt.show()


# In[14]:


# Comparer les erreurs MSE de la régression linéaire et Ridge pour chaque valeur de alpha les erreurs quadratiques moyenes (MSE)
#de la régression linéaire et de la régression Ridge pour chaquevaleur de alpha utilisée dans la régression Ridge.
plt.plot(alphas, [lr_mse]*len(alphas), label='Régression linéaire')
plt.plot(alphas, ridge_mse_values, label='Ridge')
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.title("Comparaison des erreurs MSE")
plt.legend()
plt.show()


# In[ ]:




