import pandas as pd 


df=pd.read_csv("human_epdnew_VgGtt.bed",sep="\t",names=["chrom","chormstart","chormend","name","score","strand"])
df.drop("score",axis=1,inplace=True)
print(df)