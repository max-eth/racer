import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="print all scores")
parser.add_argument("path", help="the path to the logs folder")
args = parser.parse_args()

data = defaultdict(lambda: defaultdict(list))

for run_name in os.listdir(args.path):
    config = json.load(open(args.path + "/" + run_name + "/1/config.json", 'r'))
    run = json.load(open(args.path + "/" + run_name + "/1/run.json", 'r'))
    metrics = json.load(open(args.path + "/" + run_name + "/1/metrics.json", 'r'))
    
    for metric_name in metrics:
        ex_name = run["experiment"]["name"]
        data[metric_name][run_name[:-2]].append(max(metrics[metric_name]["values"]))

for metric_name in data:
    sorted_data = sorted(data[metric_name], key=lambda x: max(data[metric_name][x]))
    for key in sorted_data:
        print(key, metric_name, "max:", max(data[metric_name][key]), "mean:", sum(data[metric_name][key])/len(data[metric_name][key]))
        print(key, metric_name, "vals:", data[metric_name[key]])
