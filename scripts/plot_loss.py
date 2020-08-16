import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import argparse
import os
import json


parser = argparse.ArgumentParser(description="plot loss curves")
parser.add_argument("path", help="the path to the logs folder")
parser.add_argument("--scalar", help="the scalar to plot", default="best_model")
parser.add_argument("--avg", help="the scalar to plot", default="avg_model")
parser.add_argument("--skip_plot", action="store_true")
parser.add_argument("--smooth", action="store_true")
parser.add_argument("--logy", help="log scale for y axis", action="store_true")

args = parser.parse_args()

c = {1: "green", 2: "orange", 3: "blue"}
for i in range(1, 4):
    metrics = json.load(open(os.path.join(args.path + "-" + str(i),  "1", "metrics.json"), "r"))
    print(i, "Best: ", max(metrics[args.scalar]["values"]))

    x1 = np.array(metrics[args.scalar]["steps"])
    y1 = np.array(metrics[args.scalar]["values"])
    x2 = np.array(metrics[args.avg]["steps"])
    y2 = np.array(metrics[args.avg]["values"])

    if args.smooth:
        plt.scatter(x1, y1, label="Repetition " + str(i), s=1)
        y = savgol_filter(y1, 51, 3)
        plt.plot(x1, y1)
    else:
        plt.plot(x1, y1, color=c[i], label="Repetition " + str(i) + " Best")
        plt.plot(x2, y2, color=c[i],linestyle='dashed', label="Repetition " + str(i) + " Average")

if args.logy:
    plt.yscale('log')

plt.xlabel("Iteration (step)")
plt.ylabel("Fitness (smoothed)" if args.smooth else "Fitness")
plt.grid()
plt.legend()
plt.show()
