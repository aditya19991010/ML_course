#Entropy calculation using breast_cancer dataset
# Split the data
# calculate the entropy of the dataset
# calculate entropy for each the category distribution based on target value for each class


import numpy as np
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import train_test_split


#Calculating Entropy of the dataset
def entropy(y):
    _, counts = np.unique(y, return_counts=True)  # Get unique values and their counts
    Pr = counts / len(y)
    E = np.sum([-1 * x * np.log2(x) for x in Pr if x > 0]) #sum all the entropy
    return E

#calculate entropy for the category
#set the parent and children info
#count the frequency of each feature based in y value
#subset the data
#calculate the entropy
def calc_ig(parent,lt_child, rt_child):
    #calculate entropy of dataset
    child = [pd.DataFrame(lt_child), pd.DataFrame(rt_child)]

    E = entropy(parent)
    lt_entropy = entropy(lt_child)
    rt_entropy = entropy(rt_child)
    print(lt_entropy)
    print(rt_entropy)
    print(len(lt_child)/len(parent))
    print(len(parent))
    wg_entropy = (len(lt_child)/len(parent)) * lt_entropy + (len(rt_child)* rt_entropy) *rt_entropy
    print(E)
    print(wg_entropy)
    IG = E - wg_entropy
    return IG




def main():
    # Data
    import numpy as np
    import pandas as pd

    # Fixed dataset ensuring each parent has both left and right children
    data = {
        "Parent": ["circle", "rectangle", "circle", "rectangle", "rectangle",
                   "circle", "rectangle", "rectangle", "circle", "rectangle",
                   "rectangle", "circle", "rectangle"],
    }

    lt_child ={"Left_Child": ["circle", "rectangle", "circle", "rectangle", "rectangle",
                   "circle", "rectangle", "circle"]
              }

    rt_child = {"Right_Child": ["rectangle", "circle", "rectangle", "circle", "rectangle",
                    "circle", "rectangle", "circle"]
                }

    # Convert to DataFrame
    df = pd.DataFrame(data)
    lt_child_df = pd.DataFrame(lt_child)
    rt_child_df = pd.DataFrame(rt_child)

    #Entropy calculation
    #recurrence/re =1 , no-recurrence/ no_re=0
    IG = calc_ig(df["Parent"], lt_child_df, rt_child_df)
    print(f"Information gain:{IG}")
    # print(infor_gain(E, entropy_values))

if __name__=="__main__":
    main()
