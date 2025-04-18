import pandas as pd 

df = pd.read_csv("../IMDB-test-generated.csv")

count = df.shape[0]
split_count = int(count/2)
df_test = df[:split_count]
df_dev = df[split_count:]

print("df_test:",df_test.shape)
print("df_dev:",df_dev.shape)
df_test.to_csv("IMDB-test-generated.csv",index=None,sep=',')
df_dev.to_csv("IMDB-dev-generated.csv",index=None,sep=',')
