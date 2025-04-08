import pandas as pd

#Creating a dictionary
def encoding_dict(df, colname):
    m,n = df.shape #assign dimention values
    encd_dict = dict() #create a dictionary
    set_var = list(set(df[colname])) #saving unique column names
    num = 0

    ##Creating a table for each value in a column
    ###iterations for setting number for each column
    for val in range(0,len(set_var)):
        num +=1
        encd_dict[f"{num}"] = set_var[val]

    ##assinging values in each column
    for item in range(0,m): #each column
        for key,value in encd_dict.items(): # fetch key-value pair
            if df[colname].iloc[item] == value: #assign numertical value
                df.at[item,colname] = key
    return df


# create 3 individual columns in concatenation with category name
# wherever the value is present put 1 else 0

def one_hot_enc(df, colname):
    set_cat = list(set(df[colname])) #unique col names in a list format
    m,n = df.shape #Dimentions

    #creating new names  and iterating through each column
    for i in range(0,len(set_cat)):
        new_col_name = f"{colname}_{set_cat[i]}" #generating a column name
        new_col = pd.DataFrame(columns=[f"{new_col_name}"])

        ## selecting each column and assigning values in the corresponding new column
        for p in range(0,m):
            if df[colname].iloc[p] == set_cat[i]:
                new_col.at[p, new_col_name] = 1
            else:
                new_col.at[p, new_col_name] = 0

        df = df.join(new_col) #joining new column
    df = df.drop(columns=[colname], axis=1) #removing the old column
    return df



def main():
    #one hot encoding
    data = {
        'Employee id': [10, 20, 15, 25, 30],
        'Gender': ['M', 'F', 'F', 'M', 'F'],
        'Remarks': ['Good', 'Nice', 'Good', 'Great', 'Nice']
    }

    df = pd.DataFrame(data)
    print(f"Orginial data: \n{df}")
    colname = 'Remarks'
    print(f"Selecting : {colname}")

    #Ordinal encoding
    print("\n", "-"*30)
    encd_dict = encoding_dict(df, colname)
    print(f"Ordinal encoding:\n{encd_dict}")

    #One hot encoding
    print("\n", "-"*30)
    new_df = one_hot_enc(df, colname)
    print(f"One hot encoding: \n{new_df}")

if __name__=="__main__":
    main()