#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#%%
raw_data = pd.read_excel('../../data/raw_data.xlsx')

#%%
sns.boxplot(raw_data['CONTRACT_TYPE'])