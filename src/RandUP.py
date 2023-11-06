"""

Sampling based reachability analysis for a tractor-trailer system

Robot model: differential driven mobile robot with passive trailer, reference point on the trailer

State variable: (x2, y2, theta1, theta2)
Control input variable: (v1, w1)

author: Yucheng Tang (@Yucheng-Tang)

RandUP: https://github.com/StanfordASL/RandUP
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from shapely.geometry import Polygon
from sklearn.decomposition import PCA
import scipy.spatial


# Define constants
v_limit = 0.2
ang_limit = 30.0
time_interval = 50
DT = 0.1

# Vehicle parameters
LENGTH = 0.72  # [m]
LENGTH_T = 0.36  # [m]
WIDTH = 0.48  # [m]
BACKTOWHEEL = 0.36  # [m]
WHEEL_LEN = 0.1  # [m]
WHEEL_WIDTH = 0.02  # [m]
TREAD = 0.2  # [m]
WB = 0.3  # [m]
ROD_LEN = 0.5  # [m]
CP_OFFSET = 0.1  # [m]

# Define control input limits
U_min = [-v_limit, np.deg2rad(-ang_limit)]
U_max = [0, np.deg2rad(ang_limit)]
# set v_max to 0 for pure backward movement

# select random control input
U_min_random = U_min
U_max_random = U_max
for i in range(time_interval - 1):
    U_min_random = np.concatenate((U_min_random, U_min), axis=None)
    U_max_random = np.concatenate((U_max_random, U_max), axis=None)

# Function to simulate state update
def f(x, u):
    new_x = x.copy()

    new_x[0] += u[0] * math.cos(x[2] - x[3]) * math.cos(x[3]) * DT
    new_x[1] += u[0] * math.cos(x[2] - x[3]) * math.sin(x[3]) * DT
    new_x[2] += u[1] * DT
    new_x[3] += (u[0] / ROD_LEN) * math.sin(x[2] - x[3]) * DT - (CP_OFFSET * u[1] * math.cos(x[2] - x[3]) / ROD_LEN) * DT
    return new_x

# Function to plot the vehicle's outline and wheels
def plot_car(x, y, yaw, length, ax, steer=0.0, cabcolor="-r", truckcolor="-k"):
    outline = np.array([[-length / 2, (length - length / 2), (length - length / 2), -length / 2, -length / 2],
                        [WIDTH / 2, WIDTH / 2, -WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    rr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD,
                          -WHEEL_WIDTH - TREAD]])

    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    ax.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    ax.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    ax.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    ax.plot(x, y, "*")

# Function to calculate intersection area between 2 Polygons
def Cal_area_2poly(data1, data2):
    poly1 = Polygon(data1).convex_hull
    poly2 = Polygon(data2).convex_hull

    print("Area of reachable set:", poly1.area, poly2.area)

    if not poly1.intersects(poly2):
        inter_area = 0
    else:
        inter_area = poly1.intersection(poly2).area

    return inter_area / poly1.area * 100

# Function to create an obstacle Polygon
def create_obstacle(vertices):
    polygon = Polygon(vertices)
    return polygon

# set parameters for RandUP
M = 10000
limit = 0
max_area = 0
angle = np.array([np.random.uniform(low=-np.deg2rad(45), high=np.deg2rad(45), size=M)])
x = np.zeros((M, 3))
x = np.concatenate((x, angle.T), axis=1)

res = np.zeros((M, 9))

for ind in range(len(x)):
    x[0] = [0, 0, 0, np.deg2rad(0)]
    # x[1] = [0, 0, 0, np.deg2rad(30)]
    # x[2] = [0, 0, 0, np.deg2rad(-45)]
    # print(ind, np.rad2deg(x[ind, 3]))

    # with random control inputs but keep same during the time interval
    ys = np.zeros((M, 4))
    # with totally random control inputs
    ys_random = np.zeros((M, 4))
    # max velocity with bang-bang angular velocity, ysu - upper_bound w, ysl - lower_bound w, ysz - w=0
    ysu = np.zeros(4)
    ysl = np.zeros(4)
    ysz = np.zeros(4)

    us = np.random.uniform(low=U_max, high=U_min, size=(M, 2))
    us_random = np.random.uniform(low=U_max_random, high=U_min_random, size=(M, 2 * time_interval))

    for t in range(time_interval):
        if t == 0:
            ysu = f(x[ind], [-v_limit, np.deg2rad(65)])
            ysl = f(x[ind], [-v_limit, -np.deg2rad(65)])
            ysz = f(x[ind], [-v_limit, 0])
            for i in range(len(us)):
                ys[i, :] = f(x[ind], us[i])
                ys_random[i, :] = f(x[ind], us_random[i, t * 2:t * 2 + 2])
        else:
            ysu = f(ysu, [-v_limit, np.deg2rad(65)])
            ysl = f(ysl, [-v_limit, -np.deg2rad(65)])
            ysz = f(ysz, [-v_limit, 0])
            for i in range(len(us)):
                ys[i, :] = f(ys[i], us[i])
                ys_random[i, :] = f(ys_random[i], us_random[i, t * 2:t * 2 + 2])

    data1 = ys[:, :2]
    data2 = np.array([[0, 0], ysu[:2], ysl[:2], ysz[:2]])

    # Use PCA to get maximum and minimum value along the principle axis (vertices calculation)
    pca = PCA(n_components=1)
    components = pca.fit_transform(data1)
    print(pca.components_)
    theta = math.atan2(pca.components_[0, 1], pca.components_[0, 0])
    print(np.rad2deg(theta))
    data3 = data1 @ np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

    if np.mean(data3[:, 0]) < 0:
        data3pnt = np.array([data3[np.argmin(data3, axis=0)[0]], data3[np.argmin(data3, axis=0)[1]],
                             data3[np.argmax(data3, axis=0)[0]], data3[np.argmax(data3, axis=0)[1]]])
    else:
        data3pnt = np.array([data3[np.argmax(data3, axis=0)[0]], data3[np.argmax(data3, axis=0)[1]],
                             data3[np.argmin(data3, axis=0)[0]], data3[np.argmin(data3, axis=0)[1]]])

    data3pnt = data3pnt @ np.array([[1.1, 0], [0, 2]])
    data3pnt = data3pnt @ np.array([[math.cos(-theta), -math.sin(-theta)], [math.sin(-theta), math.cos(-theta)]])
    data1pnt = np.array([data1[np.argmax(data3, axis=0)[0]], data1[np.argmax(data3, axis=0)[1]],
                         data1[np.argmin(data3, axis=0)[0]], data1[np.argmin(data3, axis=0)[1]]])

    area = Cal_area_2poly(data1, data3pnt)
    print("Intersection area: ", area)

    # insert an obstacle and calculate intersection area
    obstacle_vertices = [(-0.5, 0), (-0.7, 0.3), (-1, 0), (-0.7, -0.3)]
    obstacle = create_obstacle(obstacle_vertices)

    ox, oy = obstacle.exterior.xy

    # subfigure configuration
    fig = plt.figure(figsize=[8, 6])

    ax1 = fig.add_subplot(111)
    ax2 = fig.add_axes([0.6, 0.1, 0.3, 0.3])

    hull = scipy.spatial.ConvexHull(ys_random[:, :2])
    ax1.scatter(ys_random[:, 0], ys_random[:, 1], color='b')
    ax1.fill(ox, oy, color='red')
    for s in hull.simplices:
        ax1.plot(ys_random[s, 0], ys_random[s, 1], 'g')

    ax2.axis('off')
    # ax2.set_xlim([-1, 1.5])
    # ax2.set_ylim([-0.8, 0.8])

    for s in hull.simplices:
        ax2.plot(ys_random[s, 0], ys_random[s, 1], 'g')

    plot_car(x[ind, 0], x[ind, 1], x[ind, 3], LENGTH_T, ax2)

    plot_car(x[ind, 0] + np.cos(x[ind, 3]) * ROD_LEN + np.cos(x[ind, 2]) * CP_OFFSET,
             x[ind, 1] + np.sin(x[ind, 3]) * ROD_LEN + np.sin(x[ind, 2]) * CP_OFFSET, x[ind, 2], LENGTH, ax2)
    ax2.plot
    plt.show()

