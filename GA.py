# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import random
import time

#问题： 1. decode1:  count == ttime  ttime 在本环境下是一个定值 如何应用卫星？
#       2. 每个种群里的任务是否相同
#        3.
def plot_profit(profit,avg):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

 
    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Total reward")
 
    plt.title('FCFS')
  

    
    # plot curves
    p1, = host.plot(range(len(profit)), profit, label="Total Profit")
    p2, = host.plot(range(len(avg)), avg, label="Average profit")
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=1)
 
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0, 100])
    host.set_ylim([0,400])
    # par1.set_ylim([-0.1, 1.1])
 
    plt.draw()
    plt.show()
    
    
def plot_tasks(tasks,avg1):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

 
    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Total accepted tasks")
 
    plt.title('FCFS')
    # plot curves
    p1, = host.plot(range(len(tasks)), tasks, label="Total Accepetd Tasks")
    p2, = host.plot(range(len(avg1)), avg1, label="Average profit")
 
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=1)
 
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0,100])
    host.set_ylim([0,60])
    # par1.set_ylim([-0.1, 1.1])
 
    plt.draw()
    plt.show()
    
def plot_ap(ap,avg2):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window

 
    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Average Profit")
 
    plt.title('FCFS')
    # plot curves
    p1, = host.plot(range(len(ap)), ap, label="Average Profit")
    p2, = host.plot(range(len(avg2)), avg2, label="Running average profit")
 
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=1)
 
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0,100])
    host.set_ylim([0,9])
    # par1.set_ylim([-0.1, 1.1])
 
    plt.draw()
    plt.show()


task=[]
arrival=[]
code=[]
total_tasks = 100


def produce_tasks():
    task.clear()
    arrival.clear()
    for i in range(total_tasks):
        a= random.randint(0,1200)
        #a= random.randint(0,888)
        arrival.append(a)
    arrival.sort()
    for i in range(total_tasks):
        task.append([])
    storage = 5
    for i in range(total_tasks):
        task[i].append(arrival[i])
        #et= random.randint(10,15)
        et= random.randint(20,40)
        task[i].append(et)
        task[i].append(storage)
        profit = random.randint(1,10)
        task[i].append(profit)
        #task[i].append(0)
        task[i].append(arrival[i])
        #task[i].append(0)
        task[i].append(arrival[i]+et)
    return task

###初始化一个种群 popsize为种群个数，n为每个种群内的个数
def init(popsize, n):
    population = []
    for i in range(popsize):
        pop = ''
        for j in range(n):
            ###每个种群内的每个个体可以取值为0或1，0为不被选择，1为被选择
            pop = pop + str(np.random.randint(0, 2))
        population.append(pop)
    return population

#x[i] 具体是指？？？？
def decode1(x, n, w, W, tasks, T):
    tasks_list = []  # 储存被选择物体的下标集合
    TS = 5
    ###初始化时间
#    for j in range(T):
#        ttt.append(0)
    storage = 0
    value = 0
    count = 0
    #0 代表不接受， 1代表接受
    for i in range(n):

        if(x[i] == '1'):
            if (count == 0):
                tasks_list = tasks_list + [i]
                storage  = storage + w
                value = value + tasks[i][3]
                count = count+1
            else:
                length = len(tasks_list)
                #print('length',length)
                j = tasks_list[length-1]
                y = tasks[j][5]+2*TS
                #print(y)
                #print(tasks[i])
                if(tasks[i][0]>y):
                    if(storage+w>W or tasks[i][5]>T):
                        break
                    else:
                        storage  = storage + w
                        tasks_list = tasks_list + [i]
                        value = value + tasks[i][3]
    return value,tasks_list

                


def fitnessfun1(population, n, w, W, tasks, T):
    value = []  ###储存每个种群的价值
    lists = []  ###储存每个种群被选择的索引
    for i in range(len(population)):
        [f, s] = decode1(population[i], n, w, W,tasks, T)
        value.append(f)
        lists.append(s)
    return value, lists


###轮盘模型
###以每个种群的价值占总价值和的比作为轮盘的构成，价值高的则占轮盘的面积大，即该染色体生存或选择概率更大
def roulettewheel(population, value, pop_num):
    fitness_sum = []
    ###价值总和
    value_sum = sum(value)
    ###每个价值的分别占比， 总和为1
    fitness = [i / value_sum for i in value]
    ###从种群索引0开始逐渐构成一个总和为1的轮盘
    for i in range(len(population)):  ##
        if i == 0:
            fitness_sum.append(fitness[i])
        else:
            fitness_sum.append(fitness_sum[i - 1] + fitness[i])
    population_new = []
    for j in range(pop_num):  ###
        ###轮盘指针随机转
        r = np.random.uniform(0, 1)
        ###选择是哪一个种群（染色体）被选中了
        for i in range(len(fitness_sum)):  ###
            if i == 0:
                if 0 <= r <= fitness_sum[i]:
                    population_new.append(population[i])
            else:
                if fitness_sum[i - 1] <= r <= fitness_sum[i]:
                    population_new.append(population[i])
    return population_new


###交叉
def crossover(population_new, pc, ncross):
    a = int(len(population_new) / 2)
    ###选择出所有种群的双亲（所有染色体的双亲）
    parents_one = population_new[:a]
    parents_two = population_new[a:]
    ###随机每个种群（染色体的顺序）
    np.random.shuffle(parents_one)
    np.random.shuffle(parents_two)
    ###后代
    offspring = []
    for i in range(a):
        r = np.random.uniform(0, 1)
        if r <= pc:
            ###在每个种群中产生两个断点
            point1 = np.random.randint(0, (len(parents_one[i]) - 1))
            point2 = np.random.randint(point1, len(parents_one[i]))
            ###两个父代交叉产生两个后代，假如父代分别为 abc和def 则两个后代为aec和dbf
            off_one = parents_one[i][:point1] + parents_two[i][point1:point2] + parents_one[i][point2:]
            off_two = parents_two[i][:point1] + parents_one[i][point1:point2] + parents_two[i][point2:]
            ncross = ncross + 1
        else:
            off_one = parents_one[i]
            off_two = parents_two[i]
        offspring.append(off_one)
        offspring.append(off_two)
    return offspring


###变异1
###每整条染色体分别检验变异概率，如果变异，则在该染色体上产生一个需要变异的点
def mutation1(offspring, pm, nmut):
    for i in range(len(offspring)):
        r = np.random.uniform(0, 1)
        if r <= pm:
            ###随机选出一个点进行变异，如果该点是选择就变成不被选择，如果是不被选择则变成被选择
            point = np.random.randint(0, len(offspring[i]))
            if point == 0:
                if offspring[i][point] == '1':
                    offspring[i] = '0' + offspring[i][1:]
                else:
                    offspring[i] = '1' + offspring[i][1:]
            else:
                if offspring[i][point] == '1':
                    offspring[i] = offspring[i][:(point - 1)] + '0' + offspring[i][point:]
                else:
                    offspring[i] = offspring[i][:(point - 1)] + '1' + offspring[i][point:]
            nmut = nmut + 1
    return offspring


# 对每条染色体上的每个点进行变异概率检验
def mutation2(offspring, pm, nmut):
    for i in range(len(offspring)):
        for j in range(len(offspring[i])):
            r = np.random.uniform(0, 1)
            if r <= pm:
                if j == 0:
                    if offspring[i][j] == '1':
                        offspring[i] = '0' + offspring[i][1:]
                    else:
                        offspring[i] = '1' + offspring[i][1:]
                else:
                    if offspring[i][j] == '1':
                        offspring[i] = offspring[i][:(j - 1)] + '0' + offspring[i][j:]
                    else:
                        offspring[i] = offspring[i][:(j - 1)] + '1' + offspring[i][j:]
                nmut = nmut + 1 
    return offspring


def produce_value(task):
    value = []
    t = len(task)
    for i in range(t):
        value.append(task[i][3])
    return value
def produce_time(task):
    E_t = []
    t = len(task)
    for i in range(t):
        E_t.append(task[i][1])
    return E_t
# 主程序----------------------------------------------------------------------------------------------------------------------------------
# 参数设置-----------------------------------------------------------------------
def GA():
    gen = 200  # 迭代次数
    pc = 0.25  # 交叉概率
    pm = 0.02  # 变异概率
    popsize = 10  # 种群大小
    n = 100  # 任务数量,即染色体长度n
    tasks = produce_tasks()
    #print('******************tasks********************')
    #print(tasks)
    #c = [5, 7, 9, 4, 3, 5, 6, 4, 7, 1, 8, 6, 1, 7, 2, 9, 5, 3, 2, 6]  # 每个物品的价值列表
    w = 5  # 每个物品所占据的重量
    W = 250  # 存储空间
    T = 1200  # 总时间区间
    fun = 1  # 1-第一种解码方式，2-第二种解码方式（惩罚项）
    # 初始化-------------------------------------------------------------------------
    # 初始化种群（编码）
    number = 0
    value = 0
    population = init(popsize, n)
    # 适应度评价（解码）
    if fun == 1:
        value, s = fitnessfun1(population, n, w,W,tasks,T)
    # 初始化交叉个数
    ncross = 0
    # 初始化变异个数
    nmut = 0
    # 储存每代种群的最优值及其对应的个体
    t = []
    best_ind = []
    #last = []  # 储存最后一代个体的适应度值
    #realvalue = []  # 储存最后一代解码后的值
    # 循环---------------------------------------------------------------------------
    for i in range(gen):
        #print("迭代次数：")
        #print(i)
        # 交叉
        offspring_c = crossover(population, pc, ncross)
        # 变异
        # offspring_m=mutation1(offspring,pm,nmut)
        offspring_m = mutation2(offspring_c, pm, nmut)
        mixpopulation = population + offspring_m
        # 适应度函数计算
        if fun == 1:
            value, s = fitnessfun1(mixpopulation, n, w, W, tasks, T)
        # 轮盘赌选择
        population = roulettewheel(mixpopulation, value, popsize)
        # 储存当代的最优解
        result = []
        if i == gen - 1:
            if fun == 1:
                value1, s1 = fitnessfun1(population, n, w, W, tasks, T)
                #realvalue = s1
                result = value1
                #last = value1
        else:
            if fun == 1:
                value1, s1 = fitnessfun1(population, n, w, W, tasks, T)
                result = value1
        maxre = max(result)
        h = result.index(max(result))
        # 将每代的最优解加入结果种群
        t.append(maxre)
        best_ind.append(population[h])
    
    # 输出结果-----------------------------------------------------------------------
    if fun == 1:
        #best_value = max(t)
        hh = t.index(max(t))
        print(best_ind[hh])
        #print(n)
        #print(w)
        #print(W)
        #print(tasks)
        #print(T)
        f2, s2 = decode1(best_ind[hh], n, w, W, tasks, T)
        print("此次最优组合为：")
        print(s2)
        number = len(s2)
        print(len(s2))
        print("此次最优解为：")
        print(f2)
        value = f2
        print("此次最优解出现的代数：")
        print(hh)
    return number, value

episode = 100
eval_profit_list = []
avg_profit_list=[]
eval_tasks_list=[]
avg_tasks_list=[]
eval_ap_list=[]
avg_ap_list=[]
avg =0
avg_1 = 0
avg_2 = 0
jy = 1
T = 0
for i in range(episode):
    start = time.time()
    print('episode:',jy)
    jy= jy+1
    t_n,profit=GA()
    eval_profit_list.append(profit)
    avg = sum(eval_profit_list)/len(eval_profit_list)
    avg_profit_list.append(avg)        
    eval_tasks_list.append(t_n)
    avg_1 = sum(eval_tasks_list)/len(eval_tasks_list)
    avg_tasks_list.append(avg_1)
    avg_profit = profit/t_n
    eval_ap_list.append(avg_profit)
    avg_2 = sum(eval_ap_list)/len(eval_ap_list)
    avg_ap_list.append(avg_2)
    end = time.time()
    times = end -start
    T = T+times
print('Aveage Total Profit', avg)
print('Aveage accepted tasks',avg_1)
print('Aveage Profit', avg_2)
print('Average Response Time', T/episode)
plot_profit(eval_profit_list,avg_profit_list)
plot_tasks(eval_tasks_list,avg_tasks_list)
plot_ap(eval_ap_list,avg_ap_list)


