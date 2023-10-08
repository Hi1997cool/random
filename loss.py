import numpy as np

'''计算loss'''
def computer_erro_for_line_given_points(b,w,points):   #points是一系列x，y的组合
    totalError=0    #总误差
    for i in range(0,len(points)):   #对每个points点来做循环迭代
        x=points[i,0]    #对每个点取得这个点的x值
        y=points[i,1]    #取得这个点的y值
        totalError += (y-(w*x+b))**2   #实际的值与预测值之间的之差的平方和
        return totalError / float(len(points))    #加了average，计算总的points点的数量，返回的是总的average和loss

'''计算gradient'''
def step_gradient(b_current,w_current,points,learningRate):
    b_gradient=0   #梯度=斜率，相当于最小值的地方
    w_gradient=0
    N = float(len(points))
    for i in range(0,len(points)):
        x=points[i,0]   #每个点上x的值
        y=points[i,1]   #每个点上y的值
        b_gradient += -(2/N)*(y - ((w_current * x) + b_current))  #2/N是在算一个averange  loss=（WX+B - y)[负号，也由此得来]
        w_gradient += -(2/N)*x*(y - ((w_current * x) +b_current))
    new_b = b_current - (learningRate * b_current)  #根据公式，需要* 学习率
    new_w = w_current -(learningRate * w_current)   #同理
    return[new_b,new_w]    #返回一个新的w，和新的b

'''循环迭代梯度信息'''
def gradient_descent_runner(points,starting_b,starting_m,learning_rate,num_iterations):
    b=starting_b
    m=starting_m
    for i in range(num_iterations):
        b,m = step_gradient(b,m,np.array(points),learning_rate)  #b,m，（x,y),学习率
        return[b,m]   #循环无数次之后得到的比较优的解

def run():
    points = np.genformtxt("data.csv" , delimiter=",")
    learning_rate=0.001
    initial_b=0   #初试截距
    initial_m=0   #初始斜率
    num_iterations=1000  #迭代次数
    print("starting gradient descent at b = {0}, m={1}, error = {2}".
          format(initial_b,initial_m,
                 computer_erro_for_line_given_points(initial_b , initial_m , points))
          )
    [b,m] = gradient_descent_runner(points ,initial_b ,initial_m ,learning_rate ,points)
    print("After {0} initial_b = {1} , m = {2} , error = {3}".format(num_iterations , b , m))
