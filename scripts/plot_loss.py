import matplotlib.pyplot as plt
import numpy as np

import argparse
import os
import json

parser = argparse.ArgumentParser(description="plot loss curves")
parser.add_argument("path", help="the path to the logs folder")
parser.add_argument("--scalar", help="the scalar to plot", default="fitness")
parser.add_argument("--skip_plot", action="store_true")

args = parser.parse_args()

for i in range(1, 5):
    metrics = json.load(open(os.path.join(args.path + "-" + str(i),  "1", "metrics.json"), "r"))
    print(i, "Best: ", max(metrics[args.scalar]["values"]))

    plt.plot(metrics[args.scalar]["steps"], metrics[args.scalar]["values"], label=str(i))
plt.legend()
plt.show()
