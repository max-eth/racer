import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import argparse
import os
import json


parser = argparse.ArgumentParser(description="plot loss curves")
parser.add_argument("path", help="the path to the logs folder")
parser.add_argument("--scalar", help="the scalar to plot", default="fitness")
parser.add_argument("--skip_plot", action="store_true")
parser.add_argument("--smooth", action="store_true")
parser.add_argument("--logy", help="log scale for y axis", action="store_true")
parser.add_argument("--three", help="plot the first 3 runs", action="store_true")

args = parser.parse_args()


for i in range(1, 4 if args.three else 5):
    metrics = json.load(
        open(os.path.join(args.path + "-" + str(i), "1", "metrics.json"), "r")
    )
    print(i, "Best: ", max(metrics[args.scalar]["values"]))

    x = np.array(metrics[args.scalar]["steps"])
    y = np.array(metrics[args.scalar]["values"])

    if args.smooth:
        plt.scatter(x, y, label="Repetition " + str(i), s=1)
        y = savgol_filter(y, 51, 3)
        plt.plot(x, y)
    else:
        plt.plot(x, y, label="Repetition " + str(i))

if args.logy:
    plt.yscale("log")

plt.xlabel("Iteration (step)")
plt.ylabel("Fitness (smoothed)" if args.smooth else "Fitness")
plt.grid()
plt.legend()
plt.show()
