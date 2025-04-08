import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import ylabel

df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv", sep=",")

print(df.head())
print(df.columns)
print(df.info())
print(f"Count: {df.count()}")
print("Dimension: ",df.shape)
# print(df.describe())

#Check data
features = ['age', 'BMI', 'BP', 'blood_sugar', 'Gender', 'disease_score']

def hist_df(df):
    df_num = df.select_dtypes(include=['float64', 'int64'])
    df_num.hist(figsize=(16, 20), bins=60, xlabelsize=8, ylabelsize=8)

def corr_df(df):
    df_num = df.select_dtypes(include=['float64', 'int64'])
    df_corr = df_num.corr()['disease_score_fluct'][:-1]
    plt.plot(df_corr)
    plt.ylabel("disease_score_fluct")
    plt.title("Correlation plot")
    print(df_corr)


#corr plot
df_num = df.select_dtypes(include=['float64', 'int64'])
df_corr = df_num.corr()['disease_score_fluct'][:-1]
df_corr_sig = df_corr[abs(df_corr) > 0.5]

df_corr_matrix = pd.DataFrame(df_num).corr()

print(df_corr_matrix)
sns.heatmap(df_corr_matrix)
plt.show()
# df_corr_sig_col = df[df["age", "disease_score"]]


# #boxplot with gender showing negative correlation
# sns.boxplot(x=df['Gender'])
# plt.show()

# disease score fluctuation
#filter features
# df_fil = df[]

def box_plt(df):
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(x='Gender', y='disease_score_fluct', data=df)
    plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
    plt.xticks(rotation=45)

x = box_plt(df)
plt.show()

def main():
    print("Start")
    corr_df(df)
    # plt.show()
    # hist_df(df)



if __name__=="__main__":
    main()