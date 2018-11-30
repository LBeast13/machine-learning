# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 11:54:38 2018

@author: Liam
"""
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import LR_Algorithm as lra


def prepare_data(nb_points):
    """
    Prepare the dataset (training points) 
    We have a target function (only here to visualize because it is usually unknown)
    It also display the Graph
    
    Return weights, x_n, y_n, the slope and the intercept of the target function
    """
    
    #TRAINING POINTS
    #x_n, y_n = creation_dataset(nb_points)
    
    x_n, y_n = creation_dataset_NL(nb_points)
    
    return x_n, y_n


def creation_dataset(nb_points):
    """
    Generates the dataset of nbPoints (x_1X,x_1Y,y_1) using random uniform distribution.
    We use the equation of the target function to get the y_n:
    f(x1, x2) = sign(x1^2 + x2^2 - 0,6)
    
    Returns x_n and y_n, the training points
    """
    
    #DATA POINTS    
    x_n = []
    y_n = []
    
    for j in range(nb_points):
        
        x_curr = []
        
        #The w0 in order to make the matricial product
        #We will always have (1,x1,x2)
        x_curr = np.append(x_curr,1) 
        x_curr =np.append(x_curr, np.random.uniform(-1,1,2))
        x_n.append(x_curr)
        
        #We generate the output for the current x1 and x2 using the target function
        y = apply_target(x_curr)
        y_n.append(y)
    
    x_n = np.array(x_n)
    y_n = np.array(y_n)
    
    #Return an array of random indexes without duplicates
    random_indexes = rd.sample(range(nb_points), int(nb_points/10))

    #Change the sign output of 10 percent of the training points
    for index in random_indexes:
        #We change the sign of the output
        y_n[index] = - y_n[index]
   
    return x_n, y_n

def creation_dataset_NL(nb_points):
    #DATA POINTS    
    x_n = []
    y_n = []
    
    for j in range(nb_points):
        
        random_uniform = np.random.uniform(-1,1,2)
       
        x_curr = []
        
        #The w0 in order to make the matricial product
        #We will always have (1,x1,x2,x1x2,x1^2,x2^2)
        x_curr = np.append(x_curr,1) 
        x_curr = np.append(x_curr, random_uniform[0])
        x_curr = np.append(x_curr, random_uniform[1])
        x_curr = np.append(x_curr,x_curr[1]*x_curr[2])
        x_curr = np.append(x_curr,x_curr[1]**2)
        x_curr = np.append(x_curr,x_curr[2]**2)
        x_n.append(x_curr)
        
        #We generate the output for the current x1 and x2 using the target function
        y = apply_target(x_curr)
        y_n.append(y)
    
    x_n = np.array(x_n)
    y_n = np.array(y_n)
    
    #Return an array of random indexes without duplicates
    random_indexes = rd.sample(range(nb_points), int(nb_points/10))

    #Change the sign output of 10 percent of the training points
    for index in random_indexes:
        #We change the sign of the output
        y_n[index] = - y_n[index]
   
    return x_n, y_n

def apply_target(x_point):
    """
    Apply the target function 
    f(x1, x2) = sign(x1^2 + x2^2 - 0,6)
    Returns the output.
    """
    
    y = np.sign(x_point[1]**2 + x_point[2]**2 - 0.6)
    
    return y

def error_in_sample(nb_points, x_n, y_n, slope_hypot, intercept_hypot):
    
    y_n_learn = []
    
    for i in range(nb_points):
        if slope_hypot * x_n[i,1] + intercept_hypot <= x_n[i,2]:
            y_n_learn.append(1)
        else:
            y_n_learn.append(-1)

    y_n_learn = np.array(y_n_learn)
    
    E_in = compare(nb_points, y_n, y_n_learn)
    
    return E_in
    
    
def compare(nb_points, y_n, y_n_learn):
    """
    Compares the y_n targets of the training point with the y obtained using
    the learned hypothesis on the same training points.
    Used to calculate E_in and E_out
    """
    
    misclassified_count = 0 #number of misclassified points
    
    for i in range(nb_points):
        if y_n[i] != y_n_learn[i]:
            misclassified_count += 1
    
    E_in = misclassified_count/nb_points
    
    return E_in
        
def display(x_n,y_n):
     
    #GRAPH WINDOW
    axes = plt.gca()
    axes.set_xlim(-1, 1) #MIN and MAX values on X axe 
    axes.set_ylim(-1, 1) #MIN and MAX values on Y axe
    
    #DISPLAYS THE TRAINING POINTS
    #   RED : y positive
    #   BLUE : y negative
    for i in range(nb_points):
        plt.plot(x_n[i,1],x_n[i,2],'ro' if (np.sign(y_n[i]) == 1) else 'bo')
    
    plt.show()

def weight_average(w,nb_exp):
    w0_count = 0
    w1_count = 0
    w2_count = 0
    w3_count = 0
    w4_count = 0
    w5_count = 0
        
    for i in range(nb_exp):
        w0_count += w[i][0]
        w1_count += w[i][1]
        w2_count += w[i][2]
        w3_count += w[i][3]
        w4_count += w[i][4]
        w5_count += w[i][5]
    
    avg_w0 = w0_count/nb_exp
    avg_w1 = w1_count/nb_exp
    avg_w2 = w2_count/nb_exp
    avg_w3 = w3_count/nb_exp
    avg_w4 = w4_count/nb_exp
    avg_w5 = w5_count/nb_exp
    
    weight_avg = [avg_w0, avg_w1, avg_w2, avg_w3, avg_w4, avg_w5]
    
    return weight_avg

if __name__ == '__main__':
     
    #NUMBER OF TRAINING POINTS (DATASET)
    nb_points = 1000
    
    #NUMBER OF INDEPENDENT EXPERIENCES THAT WILL BE REALISED
    nb_experiences = 100
    
    all_weights = []
    sum_E_in = 0
    
    #RUN MULTIPLE EXPERIENCES
    for i in range(nb_experiences):
        
        #PREPARE THE DATASET
        x_train_pts, y_train_pts = prepare_data(nb_points)
   
        #RUN THE LINEAR REGRESSION ALGORITHM ON THE DATA
        slope_hypot, intercept_hypot, weights = lra.linear_regression(nb_points,
                                                                      x_train_pts, 
                                                                      y_train_pts)
        
        #CALCULATE E_IN AND E_OUT
        E_in_curr = error_in_sample(nb_points,x_train_pts, y_train_pts, slope_hypot,
                                    intercept_hypot)
        
        #UPDATE COUNTER
        sum_E_in += E_in_curr
        all_weights.append(weights)
        
        #DISPLAY EACH 100 EXPERIENCES
        if (i+1)%100 == 0:
            print("{} experiences already done.".format(i+1))   

    #CALCULATE THE AVERAGE
    E_in_avg = sum_E_in / nb_experiences
    weights_avg = weight_average(all_weights,nb_experiences)
    
    #DISPLAY THE LAST EXPERIMENT
    display(x_train_pts,y_train_pts)
    
    #DISPLAY THE RESULT
    print("E_in average : {}".format(E_in_avg))
    print("Weights vector : {}".format(weights))
    print("Weights average vector : {}".format(weights_avg))
    