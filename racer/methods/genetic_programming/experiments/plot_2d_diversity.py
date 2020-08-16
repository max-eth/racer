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
parser.add_argument("--logx", help="use log scale for x axis", action="store_true")
parser.add_argument("--logy", help="use log scale for y axis", action="store_true")
parser.add_argument("--mean", help="plot the mean scores", action="store_true")
parser.add_argument("--title", help="plot title")
args = parser.parse_args()

scalar = "Best ever fitness"

actual_ex_name = "genetic_programming"


for run_name in os.listdir(args.path):
    config = json.load(open(args.path + "/" + run_name + "/1/config.json", 'r'))
    run = json.load(open(args.path + "/" + run_name + "/1/run.json", 'r'))
    metrics = json.load(open(args.path + "/" + run_name + "/1/metrics.json", 'r'))
    ex_name = run["experiment"]["name"]
    if ex_name == actual_ex_name:
        data[(config["selector_params"]["tournament_size"], config["n_elitism"])].append(max(metrics[scalar]["values"]))

x, y, z = [], [], []
for (x_val, y_val), z_val in data.items():
    x.append(x_val)
    y.append(y_val)
    if args.mean:
        z.append(sum(z_val) / len(z_val))
    else:
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
if args.logy:
    ax.set_yscale('log')
if args.title is not None:
    ax.set_title(args.title)
ax.set_xlabel("toruanment_size")
ax.set_ylabel("n_elitism")
#ax.set(xlim=(-2, 2), ylim=(-2, 2))

plt.show()
