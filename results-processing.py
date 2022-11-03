import pandas as pd
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import os

font = {"size": 18}

results = pd.read_csv("results.csv")
# results = results[results["sample_rate(Hz)"].isin(
#    (24000, 12000, 6000, 3000, 1500, 750, 375))]

sample_rate_means = results.groupby("sample_rate(Hz)").mean(numeric_only=True)

sample_rate_stds = results.groupby("sample_rate(Hz)").std(numeric_only=True)

bit_width_results = results[~results["bit_width"].isin([0])]

bit_width_means = bit_width_results.groupby(
    "bit_width").mean(numeric_only=True)

bit_width_stds = bit_width_results.groupby("bit_width").std(numeric_only=True)

data_type_results = results[results["bit_width"].isin([0])]

data_type_means = data_type_results.groupby(
    "data_type").mean(numeric_only=True)

data_type_stds = data_type_results.groupby("data_type").std(numeric_only=True)

with open("processed-results.txt", "w") as file:
    file.write(
        f"Sample Rate Means:\n{str(sample_rate_means)}\nSample Rate Standard Deviations:\n{str(sample_rate_stds)}\nBit Width Means:\n{str(bit_width_means)}\nBit Width Standard Deviations:\n{str(bit_width_stds)}\nData Type Means:\n{str(data_type_means)}\nData Type Standard Deviations:\n{str(data_type_stds)}")

# Plot the decreasing means
if not os.path.exists("./figures/"):
    os.makedirs("./figures/")

# AUC ROC score plot:
sample_rate_auc_means = sample_rate_means["auc_score"]
bit_width_auc_means = bit_width_means["auc_score"]
bit_width_auc_means = bit_width_auc_means.reindex(["float64", "float32", "int32",
                                                   "int16", "int14", "int12", "int10", "int8", "int6", "int4", "int2"])
data_type_auc_means = data_type_means["auc_score"]

plt.close("all")
figure, ax = plt.subplots()
ax.plot(sample_rate_auc_means.index, sample_rate_auc_means, marker="o")
ax.set_xlim(25000, -2000)
ax.set_ylim(0.45, 1.05)
ax.set_xlabel("Sample Rate (Hz)", fontdict=font)
ax.set_ylabel("ROC AUC Score", fontdict=font)
ax.grid()
for i, label in enumerate(sample_rate_auc_means.index.to_numpy()):
    plt.annotate(label, (sample_rate_auc_means.index.to_numpy()[
                 i]-500, sample_rate_auc_means.to_numpy()[i]+0.01), fontsize="large")
plt.savefig("figures/sample_rate_auc_score.png", format="png")

# Inference time plot:
inference_means = sample_rate_means["inference_time(seconds)"]

plt.close("all")
figure, ax = plt.subplots()
ax.plot(inference_means.index, inference_means, marker="o")
ax.set_xlim(25000, -2000)
ax.set_xlabel("Sample Rate (Hz)", fontdict=font)
ax.set_ylabel("Inference Time (s)", fontdict=font)
ax.grid()
for i, label in enumerate(inference_means.index.to_numpy()):
    if i == 1:
        continue  # To not have overlapping labels
    plt.annotate(label, (inference_means.index.to_numpy()[
                 i]-500, inference_means.to_numpy()[i]+0.01), fontsize="large")
plt.savefig("figures/inference_time.png", format="png")


# Bit Width AUC plot
plt.close("all")
figure, ax = plt.subplots()
ax.plot(bit_width_auc_means.index, bit_width_auc_means, marker="o")
plt.xticks(rotation=45)
ax.set_ylim(0.975, 1.001)
ax.set_xlabel("Data Type", fontdict=font)
ax.set_ylabel("ROC AUC Score", fontdict=font)
ax.grid()
plt.savefig("figures/bit_width_auc_score.pdf", format="pdf")
