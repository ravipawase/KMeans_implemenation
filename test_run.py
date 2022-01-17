import pandas as pd
from Kmeans import Kmeans
import numpy as np


df = pd.read_csv('/home/ravindra/MKSSS_AIT/NLP_batch_Oct_2021/content_notebooks/data/utilities/cluster_5.txt',\
                 delim_whitespace=True, names = ['xcoo', 'ycoo'])
print(df.head())

clst = Kmeans(n_clusters=5, n_init=10, max_iter=1000, tol=0.01, random_state=100)
clst.fit(df, "/home/ravindra/MKSSS_AIT/NLP_batch_Oct_2021/content_notebooks/data/utilities/sample.txt")

#cluster_membership = clst.predict(df)
#print(cluster_membership)




