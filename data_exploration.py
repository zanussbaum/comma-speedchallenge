import matplotlib.pyplot as plt
import numpy as np

from collections import Counter

# Investigating speed distributions

with open("data/train.txt", 'r') as f:
    speeds = f.read().split('\n')

speeds = [float(s) for s in speeds]
count_speeds = [int(s) for s in speeds]

count = Counter()

count.update(count_speeds)

plt.bar(count.keys(), count.values())
plt.xlabel("Integer Speeds")
plt.ylabel("Count")
plt.title("Frequency of Speeds")
plt.savefig("figs/speed_freq.png")
plt.close()

plt.plot(speeds)
plt.xlabel("Time")
plt.ylabel("Speed")
plt.title("Speed vs Time")
plt.savefig("figs/speed_time.png")
plt.close()

plt.plot(speeds)
plt.xlabel("Time")
plt.ylabel("Speed")
plt.title("Log Speed vs Time")
plt.yscale('log')
plt.savefig("figs/log_speed_time.png")
plt.close()



