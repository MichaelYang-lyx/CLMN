import pandas as pd 


# df = pd.read_csv("old_test_cebab_new_concept_single.csv")
df = pd.read_csv("../test_yelp_new_concept_single.csv")
df = df[['description', 'ambiance_aspect_majority', 'food_aspect_majority', 'service_aspect_majority', 'noise_aspect_majority', 'review_majority']]

count = df.shape[0]
split_count = int(count/2)
df_test = df[:split_count]
df_dev = df[split_count:]

print("df_test:",df_test.shape)
print("df_dev:",df_dev.shape)
df_test.to_csv("test_yelp_new_concept_single.csv",index=None,sep=',')
df_dev.to_csv("dev_yelp_new_concept_single.csv",index=None,sep=',')
