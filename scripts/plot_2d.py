import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import argparse

data = defaultdict(list)

parser = argparse.ArgumentParser(description="plot 2d comparision")
parser.add_argument("path", help="the path to the logs folder")
parser.add_argument("x", help="the x axis config value")
parser.add_argument("y", help="the y axis config value")
parser.add_argument("--scalar", help="the scalar to plot", default="fitness")
parser.add_argument("--logx", help="use log scale for x axis", action="store_true")
args = parser.parse_args()


for run_name in os.listdir(args.path):
    config = json.load(open("logs/" + run_name + "/1/config.json", 'r'))
    run = json.load(open("logs/" + run_name + "/1/run.json", 'r'))
    metrics = json.load(open("logs/" + run_name + "/1/metrics.json", 'r'))
    ex_name = run["experiment"]["name"]
    if ex_name == "estool":
        data[(config[args.x], config[args.y])].append(max(metrics[args.scalar]["values"]))

x, y, z = [], [], []
for (x_val, y_val), z_val in data.items():
    x.append(x_val)
    y.append(y_val)
    z.append(max(z_val))



print(list(zip(x, y, z)))

fig, ax = plt.subplots()

# ----------
# Tricontour
# ----------
# Directly supply the unordered, irregularly spaced coordinates
# to tricontour.

ax.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
cntr2 = ax.tricontourf(x, y, z, levels=14, cmap="RdBu_r")

fig.colorbar(cntr2, ax=ax)
ax.plot(x, y, 'ko', ms=3)
ax.scatter(x, y)
if args.logx:
    ax.set_xscale('log')
ax.set_xlabel(args.x)
ax.set_ylabel(args.y)
#ax.set(xlim=(-2, 2), ylim=(-2, 2))

plt.show()
