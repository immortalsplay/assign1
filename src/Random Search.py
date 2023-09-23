import random
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import numpy as np

data = []
iteration_num = 2000

with open('assign1/cities.txt', 'r') as file:
# with open('cities.txt', 'r') as file:
    for line in file:
        values = line.strip().split(', ')
        if len(values) == 2:
            data.append([float(values[0]), float(values[1])])
# 定义 Draw 类
class Draw(object):
    def __init__(self):
        self.plt = plt
        self.set_font()

    def draw_line(self, ax, p_from, p_to):
        line1 = [(p_from[0], p_from[1]), (p_to[0], p_to[1])]
        (line1_xs, line1_ys) = zip(*line1)
        ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='blue'))    

    def draw_points(self, ax, pointx, pointy):
        ax.plot(pointx, pointy, 'ro', markersize=6)

    def set_xybound(self, ax, x_bd, y_bd):
        ax.axis([x_bd[0], x_bd[1], y_bd[0], y_bd[1]])

    def draw_text(self, ax, x, y, text, size=8):
        ax.text(x, y, text, fontsize=size)

    def set_font(self, ft_style='SimHei'):
        plt.rcParams['font.sans-serif'] = [ft_style]  # 用来正常显示中文标签

    def draw_citys_way(self, best_gen, citys, bound_x, bound_y):
        fig, ax = plt.subplots()
        self.set_xybound(ax, bound_x, bound_y)
        for i in range(len(best_gen) - 1):
            best_i = best_gen[i]
            next_best_i = best_gen[i + 1]
            best_icity = citys[best_i]
            next_best_icity = citys[next_best_i]
            self.draw_line(ax, best_icity, next_best_icity)
        
        start = citys[best_gen[0]]
        end = citys[best_gen[-1]]
        self.draw_line(ax, end, start)
        return ax  # 返回 ax 对象以供后续使用


# 初始化 Draw 类实例
tsp_draw = Draw()

# 生成 100 个随机城市的坐标
num_cities = 100

# def generate_random_cities(num_cities, x_range=(0, 10), y_range=(0, 10)):
#     cities = [(random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1])) for _ in range(num_cities)]
#     return cities

# cities = generate_random_cities(num_cities)

cities = data
# print('cities',cities)

# 绘制城市坐标
city_x, city_y = zip(*cities)

# 计算两个城市之间的距离
def distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# 计算路径的总长度
def total_distance(path):
    total = 0
    for i in range(len(path) - 1):
        total += distance(cities[path[i]], cities[path[i + 1]])
    total += distance(cities[path[-1]], cities[path[0]])  # 回到起始城市
    return total

def generate_neighbors(best_sol, num_neighbors=10):
    neighbors = []
    n = len(best_sol)    
    for _ in range(num_neighbors):
        neighbor = best_sol.copy()
        i, j = random.sample(range(n), 2)  # 随机选择两个不同的索引
        neighbor[i], neighbor[j] = best_sol[j], best_sol[i]  # 交换两个城市
        neighbors.append(neighbor)

    return neighbors

def parallel_hill_climbing(iteration_num=iteration_num, pool_size=4):
    all_runs_costs_at_intervals = []
    n = 5 # Number of runs
    for run in range(n):  # Run the whole process 5 times
        time_start = time.time()
        best_sol = list(range(len(cities)))
        best_cost = total_distance(best_sol)
        
        costs = []
        interval_costs = []
        
        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            for i in range(iteration_num):
                
                neighbors = generate_neighbors(best_sol)
                
                future_to_neighbor = {executor.submit(total_distance, neighbor): neighbor for neighbor in neighbors}

                next_sol = None
                next_cost = best_cost
                all_costs = []
                
                for future in concurrent.futures.as_completed(future_to_neighbor):
                    cost = future.result()
                    all_costs.append(cost)
                    neighbor = future_to_neighbor[future]
                    
                    if cost < next_cost:
                        next_sol = neighbor
                        next_cost = cost
                
                if next_sol is not None:
                    best_sol = next_sol
                    best_cost = next_cost

                costs.append(best_cost)
                
                # Record costs every 100 iterations
                if i % 100 == 0:
                    fitness = 1 / (best_cost + 1e-6)
                    interval_costs.append(fitness)
        
        all_runs_costs_at_intervals.append(interval_costs)
        
        
        time_end = time.time()
        print(f"Run {run+1} Time Cost: {time_end - time_start}s")
        print("shortest distance：", best_cost)

    # Calculate average, min, and max at each interval
    avg_costs = np.mean(all_runs_costs_at_intervals, axis=0)
    std_costs = np.std(all_runs_costs_at_intervals, axis=0)
    min_costs = avg_costs - (std_costs / (2 * np.sqrt(n)))
    max_costs = avg_costs + (std_costs / (2 * np.sqrt(n)))

    with open('Hill_short_avg_costs.txt', 'w') as f:
        for param1, param2 in zip(avg_costs, std_costs):
            f.write(f"{param1} {param2}\n")
    # Plotting
    x = np.arange(0, len(avg_costs)) * 100
    plt.plot(x, avg_costs, label='Average Cost')
    plt.errorbar(x, avg_costs, yerr=[avg_costs - min_costs, max_costs - avg_costs], fmt='o', label='Error bars')
    
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Average Cost Over Iterations with Error Bars')
    plt.legend()
    plt.show()

    # 画图部分
    city_x, city_y = zip(*[cities[i] for i in best_sol])
    plt.figure(3)
    plt.scatter(city_x, city_y, c='red')  # 绘制城市点
    plt.plot(city_x + (city_x[0],), city_y + (city_y[0],), c='blue')  # 绘制路径
    for i, txt in enumerate(best_sol):
        plt.annotate(txt, (city_x[i], city_y[i]))  # 标注城市序号

    # 添加标题和标签
    plt.title('GA Best Path Found')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.show()


    return best_sol, best_cost

# def parallel_hill_climbing(iteration_num=1000, pool_size=4):
    all_runs_costs_at_intervals = []
    n = 5 # Number of runs
    for run in range(n):  # Run the whole process 5 times
        time_start = time.time()
        best_sol = list(range(len(cities)))
        best_cost = total_distance(best_sol)
        
        costs = []
        interval_costs = []
        
        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            for i in range(iteration_num):
                
                neighbors = generate_neighbors(best_sol)
                
                future_to_neighbor = {executor.submit(total_distance, neighbor): neighbor for neighbor in neighbors}

                next_sol = None
                next_cost = best_cost
                all_costs = []
                
                for future in concurrent.futures.as_completed(future_to_neighbor):
                    cost = future.result()
                    all_costs.append(cost)
                    neighbor = future_to_neighbor[future]
                    
                    if cost < next_cost:
                        next_sol = neighbor
                        next_cost = cost
                
                if next_sol is not None:
                    best_sol = next_sol
                    best_cost = next_cost

                costs.append(best_cost)
                
                # Record costs every 100 iterations
                if i % 100 == 0:
                    interval_costs.append(best_cost)
        
        all_runs_costs_at_intervals.append(interval_costs)
        
        
        time_end = time.time()
        print(f"Run {run+1} Time Cost: {time_end - time_start}s")
        
    # Calculate average, min, and max at each interval
    avg_costs = np.mean(all_runs_costs_at_intervals, axis=0)
    std_costs = np.std(all_runs_costs_at_intervals, axis=0)
    min_costs = avg_costs - (std_costs / (2 * np.sqrt(n)))
    max_costs = avg_costs + (std_costs / (2 * np.sqrt(n)))
    
    # Plotting
    x = np.arange(0, len(avg_costs)) * 100
    plt.plot(x, avg_costs, label='Average Cost')
    plt.errorbar(x, avg_costs, yerr=[avg_costs - min_costs, max_costs - avg_costs], fmt='o', label='Error bars')
    
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Average Cost Over Iterations with Error Bars')
    plt.legend()
    plt.show()


# 随机搜索算法的实现
def randomoptimize():
    time_start = time.time() #timer
    best_sol = list(range(len(cities)))
    random.shuffle(best_sol)

    bestcost = total_distance(best_sol)
    
    # iteration_lengths = [(0, bestcost)]  # 用于记录每次迭代的长度
    iteration_lengths = []
    for i in range(iteration_num):
        sol = list(range(len(cities)))
        random.shuffle(sol)

        newcost = total_distance(sol)
        
        if newcost < bestcost:
            bestcost = newcost
            best_sol = sol
        # fitness = 1 / (bestcost + 1e-6)
        # if fitness > 1.0:
        #     fitness = 0
        iteration_lengths.append((i, bestcost))  # 记录当前迭代的长度
        
    time_end = time.time()
    print("random search Time Cost："+str((time_end - time_start))+"s")
    # print("best order：", best_sol)
    print("shortest distance：", bestcost)
    

    #save file
    # str_avg_fitness_list = [str(f) + '\n' for f in iteration_lengths]
    # with open('Random_long_lengths.txt', 'w') as f:
        # f.writelines(str_avg_fitness_list)

    # 绘制问题长度随迭代次数的变化图表
    plt.figure(1)
    plt.xlabel("iteration number")
    plt.ylabel("fitness")
    plt.title("Variation of TSP Problem Length with the Number of Iterations")
    iterations, lengths = zip(*iteration_lengths)
    plt.plot(iterations, lengths)
    plt.show()

    # 绘制最佳路径
    ax = tsp_draw.draw_citys_way(best_sol, cities, [0, math.sqrt(1)], [0, math.sqrt(1)])
    tsp_draw.draw_points(ax, city_x, city_y)  # 用返回的 ax 对象绘制点
    plt.show()

# 调用随机搜索算法来解决 TSP 问题
if __name__ == '__main__':
    # randomoptimize()

    parallel_hill_climbing()


