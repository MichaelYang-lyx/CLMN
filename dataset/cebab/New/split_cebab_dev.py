import pandas as pd 
from datasets import load_dataset

CEBaB = load_dataset("CEBaB/CEBaB")
#print("CEBaB:",CEBaB)
# df_train = CEBaB["train_exclusive"]

df_test = pd.DataFrame(CEBaB["test"]) #(1689,24)
df_val = pd.DataFrame(CEBaB["validation"]) #(1689,24)
df_train = pd.read_csv("train_cebab_new_concept_single.csv")  #(1755, 15)
print("df_train:",df_train.shape)

df_train_columns = [column for column in df_train]
df_test_columns = [column for column in df_test]
df_columns = list(set(df_train_columns) & set(df_test_columns))  
#['ambiance_aspect_majority', 'description', 'food_aspect_majority', 'service_aspect_majority', 'noise_aspect_majority', 'review_majority']
# print("df_train_columns:",df_train_columns)
# print("df_test_columns:",df_test_columns)
# print("df_columns:",df_columns)

df_test = df_test[df_columns]
df_val = df_val[df_columns]
print("df_test:",df_test.shape) #(1689,6)
print("df_val:",df_val.shape) #(1673,6)

df_test.to_csv("test_cebab_new_concept_single.csv",index=None,sep=',')
df_val.to_csv("dev_cebab_new_concept_single.csv",index=None,sep=',')
