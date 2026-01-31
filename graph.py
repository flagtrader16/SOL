import pandas as pd 
from matplotlib import pyplot as plt 

def draw(df,path):
 df['EMA_21'] = df['close'].ewm(span=21,adjust=False).mean()
 df['EMA_50'] = df['close'].ewm(span=50,adjust=False).mean()   

 df = df.tail(300)
 plt.figure(figsize=(15,6))  
  
 for s in range(2):  
    subset = df[df["state"] == s]  
    plt.scatter(  
        subset["timestamp"],  
        subset["close"],  
        s=6,  
        label=f"State {s}"  
    )  
  
 plt.plot(df["timestamp"], df["close"], color="black", alpha=0.3)  
 plt.plot(df['timestamp'], df['EMA_21'],c='blue',alpha=0.4,label='EMA_21')
 plt.plot(df['timestamp'], df['EMA_50'],c='r',alpha=0.5,label='EMA_50')
 plt.legend()  
 plt.title("Causal HMM Regimes (Forward Filter Only)")  
 plt.grid(alpha=0.3)  
 plt.savefig(path)
 return path
