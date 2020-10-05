import numpy as np

import argparse
import os
import json


parser = argparse.ArgumentParser(
    description="count the number of iterations until 900 was reached"
)
parser.add_argument("path", help="the path to the logs folder")
parser.add_argument("--scalar", help="the scalar to measure", default="fitness")
args = parser.parse_args()

vals = []
steps = []

for i in range(1, 5):
    metrics = json.load(
        open(os.path.join(args.path + "-" + str(i), "1", "metrics.json"), "r")
    )
    best = max(metrics[args.scalar]["values"])
    print(i, "Best: ", best)
    vals.append(best)

    zipped = list(zip(metrics[args.scalar]["steps"], metrics[args.scalar]["values"]))
    zipped = sorted(zipped, key=lambda x: x[0])
    for step, val in zipped:
        if val > 900:
            print("reached {} at step {}".format(val, step))
            steps.append(step)
            break
    else:
        print("Didn't reach 900 :(")

print("Mean: ", sum(vals) / len(vals))
print("Mean steps till 900: ", sum(steps) / len(steps))
