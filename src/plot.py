import matplotlib.pyplot as plt
import numpy as np

data_list1 = []
with open('Random_short_lengths.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        data_list1.append(eval(line.strip()))

iterations, lengths = zip(*data_list1)

n=5
data_list2 = []
interval = 100

with open('GA_short_avg_costs.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        a, b = map(float, line.strip().split())
        # a = a*300000
        data_list2.append((a, b))

avg_costs, std_costs= zip(*data_list2)
min_costs = avg_costs - (std_costs / (2 * np.sqrt(n)))
max_costs = avg_costs + (std_costs / (2 * np.sqrt(n)))

# 绘图
data_list3 = []

with open('Hill_short_avg_costs.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        a, b = map(float, line.strip().split())
        a=a/2
        data_list3.append((a, b))

avgcosts, stdcosts= zip(*data_list3)
mincosts = avgcosts - (stdcosts / (2 * np.sqrt(n)))
maxcosts = avgcosts + (stdcosts / (2 * np.sqrt(n)))



plt.figure(1)
x1 = np.arange(0, len(avgcosts)) * interval  # 这里的间隔需要和你实际的间隔相匹配

plt.plot(x1, avgcosts, label='Hill Average Cost')
plt.errorbar(x1, avgcosts, yerr=[avgcosts - mincosts, maxcosts - avgcosts], fmt='o')


x2 = np.arange(0, len(avg_costs)) * interval  # 这里的间隔需要和你实际的间隔相匹配
# avg_costs =1/avg_costs
plt.plot(x2, avg_costs, label='GA Average Cost')
plt.errorbar(x2, avg_costs, yerr=[avg_costs - min_costs, max_costs - avg_costs], fmt='o')

# plt.xlabel('Iteration')
# plt.ylabel('Cost')
# plt.title('Average Cost Over Iterations with Error Bars')
# lengths = 1/lengths

plt.plot(iterations, lengths, label='Random Average Cost')
plt.legend()
plt.xlabel("iteration number")
plt.ylabel("fitness")
plt.title("Variation of TSP Problem Length with the Number of Iterations")

plt.show()
