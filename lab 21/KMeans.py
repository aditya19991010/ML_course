import ISLP
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

#data preparation
df = ISLP.load_data("Weekly")
y = df["Direction"]
X = df.drop("Direction", axis="columns")


#Transformation
X_scaled =StandardScaler().fit_transform(X)
y_enc = LabelEncoder().fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.33, random_state=42)

model = KMeans(n_clusters=4,random_state=42,n_init=30)
model.fit(X_train,y_train)
#print(model.labels_)
y_pred =model.predict(X_test)


cls_report = classification_report(y_test,y_pred,zero_division=False)
print("\t\t\t\t----Classification report----\n",cls_report)



max_k=4 #add number of clusters to create
data = X_scaled # feature training data

num_plots = min(max_k, 1 * 3)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes = axes.flatten()

'''
Inertia measures how well the dataset is clustered by K-means
'''
inertias = []

for k in range(1, max_k + 1):
    model = KMeans(n_clusters=k, random_state=42, n_init=30)
    model.fit(data)
    inertias.append(model.inertia_)

    if k <= num_plots and data.shape[1] >= 2:
        axes[k - 1].scatter(data[:, 0], data[:, 1], c=model.labels_, cmap='viridis', s=30)
        axes[k - 1].set_title(f"K={k}")
        axes[k - 1].axis('off')
        axes[k - 1].grid()

for j in range(num_plots, len(axes)):
    axes[j].axis('off')

fig.suptitle("K-Means Clustering Results for Different K")
plt.tight_layout()
plt.show()



plt.plot(range(1, max_k + 1), inertias, marker='*', markersize=10, linestyle='--', color='red')
plt.grid()
plt.title("Inertia with each cluster K")
plt.show()