import torch
import pandas as pd
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from random import shuffle


l=[1,2,3,4,5]
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
token = tokenizer._convert_token_to_id('我')
print(token)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use("ggplot")
mpl.rcParams["lines.markerfacecolor"]="None"
mpl.rcParams["axes.titlecolor"]="#555555"


df_sst=pd.read_csv("stsa2_accuracy.csv",names=["percent","orig","CBERT"])
df_TREC=pd.read_csv("TREC_accuracy.csv",names=["percent","orig","CBERT"])

x=df_sst["percent"].unique()
sst_ori=[]
sst_aug=[]
TREC_ori=[]
TREC_aug=[]

for i in x:
    df=df_sst[df_sst["percent"]==i]
    sst_ori.append(df["orig"].mean())
    sst_aug.append(df["CBERT"].mean())

    df=df_TREC[df_TREC["percent"]==i]
    TREC_ori.append(df["orig"].mean())
    TREC_aug.append(df["CBERT"].mean())


plt.figure(figsize=(7,5))
plt.plot(x,sst_ori,label="original",marker='o',)
plt.plot(x,sst_aug,label="CBERT",marker='*',markersize=10,color="#988ed0")

plt.title("SST2")
plt.xlabel("Percentage of Data")
plt.ylabel("Test Accuracy")
plt.xticks([0.01,0.1, 0.2, 0.6, 1])


plt.legend(loc='lower right')
plt.grid(True)
plt.savefig("1",dpi=250)

plt.figure(figsize=(7,5))
plt.plot(x,TREC_ori,label="original",marker='o',)
plt.plot(x,TREC_aug,label="CBERT",marker='*',markersize=10,color="#988ed0")

plt.title("TREC")
plt.xlabel("Percentage of Data")
plt.ylabel("Test Accuracy")
plt.xticks([0.01,0.1, 0.2, 0.6, 1])


plt.legend(loc='lower right')
plt.grid(True)

plt.savefig("2",dpi=250)
