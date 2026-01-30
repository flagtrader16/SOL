import pandas as pd 
from matplotlib import pyplot as plt 

def draw(df_show,path="HMM_Regime.png"):
 
 plt.figure(figsize=(15,6))  
  
 for s in range(N_STATES):  
    subset = df[df["state"] == s]  
    plt.scatter(  
        subset["timestamp"],  
        subset["close"],  
        s=6,  
        label=f"State {s}"  
    )  
  
 plt.plot(df["timestamp"], df["close"], color="black", alpha=0.3)  
 plt.legend()  
 plt.title("Causal HMM Regimes (Forward Filter Only)")  
 plt.grid(alpha=0.3)  
 plt.savefig(path)
 return path
