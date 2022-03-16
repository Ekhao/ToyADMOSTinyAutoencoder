import pandas as pd
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import os

font = {"size":18}

results = pd.read_csv("results.csv")
results = results.loc[results["sample_rate(Hz)"].isin((24000,12000,6000,3000,1500,750,375))]

means = results.groupby("sample_rate(Hz)").mean()

stds = results.groupby("sample_rate(Hz)").std()

with open("processed-results.txt", "w") as file:
    file.write(f"Means:\n{str(means)}\nStandard Deviations:\n{str(stds)}")

# Plot the decreasing means
if not os.path.exists("./figures/"):
    os.makedirs("./figures/")

# AUC ROC score plot:
auc_means = means["auc_score"]

plt.close("all")
figure, ax = plt.subplots()
ax.plot(auc_means.index, auc_means, marker = "o")
ax.set_xlim(25000,-2000)
ax.set_ylim(0.45,1.05)
ax.set_xlabel("Sample Rate (Hz)", fontdict = font)
ax.set_ylabel("ROC AUC Score", fontdict = font)
ax.grid()
for i, label in enumerate(auc_means.index.to_numpy()):
    plt.annotate(label, (auc_means.index.to_numpy()[i]-500, auc_means.to_numpy()[i]+0.01), fontsize = "large")
plt.savefig("figures/auc_score.eps", format="eps")

# Inference time plot:
inference_means = means["inference_time(seconds)"]

plt.close("all")
figure, ax = plt.subplots()
ax.plot(inference_means.index, inference_means, marker = "o")
ax.set_xlim(25000,-2000)
ax.set_xlabel("Sample Rate (Hz)", fontdict = font)
ax.set_ylabel("Inference Time (s)", fontdict = font)
ax.grid()
for i, label in enumerate(inference_means.index.to_numpy()):
    if i == 1:
        continue #To not have overlapping labels
    plt.annotate(label, (inference_means.index.to_numpy()[i]-500, inference_means.to_numpy()[i]+0.01), fontsize = "large")
plt.savefig("figures/inference_time.eps", format="eps")

#print(results)
# Calculate confidence intervals maybe. But then we have to make sure that the data is normally distributed.
#print(st.norm.interval(0.95,loc=0.971900,scale=0.013336/np.sqrt(5)))