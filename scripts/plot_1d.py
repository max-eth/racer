import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import argparse

data = defaultdict(list)

parser = argparse.ArgumentParser(description="plot 1d comparision")
parser.add_argument("path", help="the path to the logs folder")
parser.add_argument("ex_name", help="experiment name")
parser.add_argument("x", help="the x axis config value")
parser.add_argument("--scalar", help="the scalar to plot", default="fitness")
parser.add_argument("--logx", help="use log scale for x axis", action="store_true")
parser.add_argument("--logy", help="use log scale for y axis", action="store_true")
parser.add_argument("--mean", help="plot the mean scores", action="store_true")
parser.add_argument("--title", help="plot title")
args = parser.parse_args()


for run_name in os.listdir(args.path):
    config = json.load(open(args.path + "/" + run_name + "/1/config.json", 'r'))
    run = json.load(open(args.path + "/" + run_name + "/1/run.json", 'r'))
    metrics = json.load(open(args.path + "/" + run_name + "/1/metrics.json", 'r'))
    ex_name = run["experiment"]["name"]
    if ex_name == args.ex_name:
        data[config[args.x]].append(max(metrics[args.scalar]["values"]))

x, y = [], []
for x_val, y_val in data.items():
    x.append(x_val)
    if args.mean:
        y.append(sum(y_val) / len(y_val))
    else:
        y.append(max(y_val))
    num_runs = len(y_val)



print(list(zip(x, y)))

fig, ax = plt.subplots()

zipped = list(zip(x, y))
sorted_values = sorted(zipped, key=lambda x: x[0])
ax.plot(*zip(*sorted_values))
if args.logx:
    ax.set_xscale('log')
if args.logy:
    ax.set_yscale('log')
if args.title is not None:
    ax.set_title(args.title)
ax.grid()
ax.set_xlabel(args.x.capitalize())
ax.set_ylabel(("Mean" if args.mean else "Max") + " fitness across " + str(num_runs) + " runs")
#ax.set(xlim=(-2, 2), ylim=(-2, 2))

plt.show()
