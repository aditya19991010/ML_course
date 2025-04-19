import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.datasets import get_rdataset
import matplotlib.pyplot as plt

df = get_rdataset("USArrests").data
print(df.head())

scale_model = StandardScaler(with_std=True, with_mean=True)
US_scaled = scale_model.fit_transform(df)

#PCA
pca = PCA()
scores = pca.fit_transform(US_scaled)
print(scores)

print("\nComponents: ",pca.components_)
print("\nExplained variance: ",pca.explained_variance_)
fig,axes = plt.subplots(1,3,figsize=(18,5))

axes[0].scatter(scores[:,0],scores[:,1])
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")


#creating arrow

for k in range(pca.components_.shape[1]):
    axes[0].arrow(0,0, pca.components_[0,k],pca.components_[1,k], color='g',head_width=0.05)
    axes[0].text(pca.components_[0,k]*1.2,pca.components_[1,k]*1.2, df.columns[k], color='k')

#plt.show()
ticks = np.arange(1,len(pca.components_) +1)

axes[1].plot(ticks, pca.explained_variance_ratio_, marker="*")
axes[1].set_xticks(ticks)
axes[1].grid()
axes[1].set_title("Explained variable per PCA")
axes[1].set_xlabel("PCs")
axes[1].set_ylabel("Variance")


axes[2].plot(ticks, pca.explained_variance_ratio_.cumsum(), marker="*", markersize=4)
axes[2].bar(ticks, pca.explained_variance_ratio_.cumsum(), color="violet")
axes[2].set_xticks(ticks)
axes[2].grid()
axes[2].set_title("Scree plot")
axes[2].set_ylabel("Variance")
axes[2].set_xlabel("PCs")


plt.show()
