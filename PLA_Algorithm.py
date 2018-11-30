# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 12:24:52 2018

This is an implementation of the Perceptron Learning Algorithm (PLA)
used for classification (binary in this case).
We use X = [0,1]x[0,1] in order to visualize it using plots

@author: Liam BETTE
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rd


def prepare_data(nb_points):
    """
    Prepare the dataset (training points) 
    We have a target function (only here to visualize because it is usually unknown)
    It also display the Graph
    
    Return weights, x_n, y_n, the slope and the intercept of the target function
    """
    
    #TARGET FUNCTION
    slope_target, intercept_target = target_function_generation()
    
    #TRAINING POINTS
    x_n, y_n = creation_dataset(nb_points, slope_target, intercept_target)
    
    return x_n, y_n, slope_target, intercept_target

def target_function_generation():
    """ 
    Generate the target function with 2 random uniform points and using the 
    slope and the intercept of the line.
    
    Returns a (the slope) and b(the intercept)
    """

    #Random points in [-1,1]x[-1,1]
    point_1 = [np.random.uniform(-1,1,1),np.random.uniform(-1,1,1)]
    point_2 = [np.random.uniform(-1,1,1),np.random.uniform(-1,1,1)]

    #TARGET FUNCTION SLOPE AND INTERCEPT
    a = (point_2[1]-point_1[1])/(point_2[0]-point_1[0]) #slope
    b = point_1[1]-(a*point_1[0]) #intercept 
    
    return a,b


def creation_dataset(nb_points, slope, intercept):
    """
    Generates the dataset of nbPoints (x_1X,x_1Y,y_1) using random uniform distribution.
    We use the equation of the target function to get the y_n:
    if the point is under the line is -1, and 1 if it is over.
    
    Returns x_n and y_n, the training points
    """
    
    #DATA POINTS    
    x_n = []
    
    for j in range(nb_points):
        x_curr = np.random.uniform(-1,1,2)
        
        #The w0 in order to make the matricial product
        #We will always have (x1,x2,1)
        x_curr = np.append(x_curr,1) 
        x_n.append(x_curr)

    x_n = np.array(x_n)
    
    #CALCULATING THE Y of the X DATA POINTS
    y_n = []

    for i in range(nb_points):
        if slope*x_n[i,0] + intercept <= x_n[i,1]:
            y_n.append(1)
        else:
            y_n.append(-1)

    y_n = np.array(y_n)
   
    return(x_n,y_n)
 
    
def is_misclassified(weights,x_curr,y_curr):
    """
    Tests if the point x_curr is Misclassified by checking signs.
    We check if the sign(weights*x_curr) != sign(y_curr)
    If it is so the point is misclassified
    """
    
    if np.sign(np.dot(weights,x_curr)) != np.sign(y_curr):
        return True
    else:
        return False


def probability_error(nb_test_points, weights, slope_target, 
                      intercept_target):
    """
    Approximate the probability of getting f(x) != g(x) by testing g(x) on random points
    
    Returns this probability.
    """
    
    #Test dataset
    x_n_test_points, y_n_test_points = creation_dataset(nb_test_points,
                                                       slope_target,
                                                       intercept_target) 
    
    #Number of misclasified points in the Test dataset according g(x)
    nb_misclas = 0 
    
    for i in range(nb_test_points):
        if is_misclassified(weights,x_n_test_points[i],y_n_test_points[i]): 
            nb_misclas += 1
          
    proba = nb_misclas / nb_test_points #the probability of getting f != g
   
    return proba


def update_weights(x_n, y_n, misclassified_point_indexes, weights):
    """
    Update the weights table by using a random misclassified point.
    It uses the formula : w = w + x_n*y_n 
    
    Returns the weights array.
    """
    
    nb_misclassified_points = len(misclassified_point_indexes)
    
    #Pick a random misclassified point
    random_index = misclassified_point_indexes[rd.randint(0,nb_misclassified_points-1)]
        
    weights[0] = weights[0] + x_n[random_index,0]*y_n[random_index]
    weights[1] = weights[1] + x_n[random_index,1]*y_n[random_index]
    weights[2] = weights[2] + y_n[random_index]
    
    return weights

 
def display_experience(nb_points, x_n, y_n, slope_hypot, intercept_hypot,
                       slope_target, intercept_target, weights):
    """
    Displays the experience: 
    target function
    hypothesis learned
    training points
    """
    
    #GRAPH WINDOW
    axes = plt.gca()
    axes.set_xlim(-1, 1) #MIN and MAX values on X axe 
    axes.set_ylim(-1, 1) #MIN and MAX values on Y axe
    
    #DISPLAYS THE TRAINING POINTS
    #   RED : y = +1 or 0
    #   YELLOW : y = -1
    for i in range(nb_points):
        plt.plot(x_n[i,0],x_n[i,1],'ro' if (y_n[i] >= 1) else 'bo')
            
    #POINTS TO DISPLAY THE TARGET FUNCTION
    x_target = np.linspace(-1.,1.,100) #100 random points between -1 and 1
    y_Target = slope_target * x_target + intercept_target #y associated to the 100 points
    
    plt.plot(x_target,y_Target,'g-') #Display the line of the target function
    
    
    #DISPLAY OF THE HYPOTHESIS (LEARNING)
    x_hypot = np.linspace(-1.,1.,100)

    #y = ax + b, a is slope and b is intercept
    y_hypot = (slope_hypot * x_hypot) + intercept_hypot
    plt.plot(x_hypot, y_hypot,'y-')
    
    plt.show()


def perceptron(nb_points, x_train, y_train, weights, nb_test_points, 
               slope_target, intercept_target):
    """
    The Perceptron Learning Algorithm (PLA)
    
    nb_points are the number of training points 
    x_train, y_train are the training points
    
    Returns:
    The slope of the hypothesis
    The intercept of the hypothesis
    The probability average
    The number of iterations made to converge
    """
    
    #THE INDEXES OF THE MISCLASSIFIED POINTS
    misclas_indexes = []
    
    #THE NUMBER OF ITERATION DONE
    nb_iteration = 0
    
    all_classified = False
    
    #LOOP WHILE THERE ARE STILL MISCLASSIFIED POINTS    
    while all_classified == False:

        #FILL THE ARRAY WITH THE INDEX OF THE MISCLASSIFIED POINTS
        for i in range(nb_points):
            if is_misclassified(weights ,x_train[i],y_train[i]):
                misclas_indexes.append(i) #Add the index if the point is misclassified
                
        #Conversion in a numpy array
        misclas_indexes = np.array(misclas_indexes)
            
       
        #IF THERE ARE MISCLASSIFIED POINTS
        if len(misclas_indexes) > 0:
            
            #UPDATE THE WEIGHTS
            weights = update_weights(x_train, y_train, misclas_indexes, weights)
            
            #RESET THE ARRAY OF MISCLASSIFIED POINTS
            misclas_indexes = []
        
        else:
            all_classified = True
           
        nb_iteration += 1
        
    #We have the equation: w[2] + w[1]*y + w[0]*x = 0
    slope_hypot = -(weights[0])/(weights[1])  
    intercept_hypot = -weights[2]/weights[1]
    
    #THE PROBABILITY OF GETTING F!=G
    proba = probability_error(nb_test_points, weights, slope_target,
                              intercept_target)
    
    return slope_hypot, intercept_hypot, proba, nb_iteration
        

if __name__ == '__main__':
 
    #NUMBER OF TRAINING POINTS (DATASET)
    nb_points = 10 
    
    #NUMBER OF TEST POINTS TO CALCULATE THE PROBABILITY
    nb_test_points = 100
    
    #NUMBER OF INDEPENDENT EXPERIENCES THAT WILL BE REALISED
    nb_experiences = 1
    

    iteration_count = 0
    proba_count = 0
    
    #RUN MULTIPLE EXPERIENCES
    for i in range(nb_experiences):
        
        #INITIAL WEIGHTS
        weights = [0.0, 0.0, 0.0]
        
        #PREPARE THE DATASET
        x_train_pts, y_train_pts, slope_target, intercept_target = prepare_data(nb_points)
   
        #RUN THE PERCEPTRON LEARNING ALGORITHM ON THE DATA
        slope_hypot, intercept_hypot, proba_misclas, nb_iter = perceptron(nb_points, 
                                                                          x_train_pts, 
                                                                          y_train_pts, 
                                                                          weights,
                                                                          nb_test_points,
                                                                          slope_target,
                                                                          intercept_target)
        
        
        #DISPLAY EACH 100 EXPERIENCES
        if (i+1)%100 == 0:
            print("{} experiences already done.".format(i+1))
            
        iteration_count += nb_iter
        proba_count += proba_misclas
    
    #AVERAGES    
    iter_avg = iteration_count/nb_experiences
    proba_avg = proba_count/nb_experiences 
    
    #DISPLAY THE FINAL HYPOTHESIS
    display_experience(nb_points, x_train_pts, y_train_pts, slope_hypot, 
                       intercept_hypot, slope_target, intercept_target,
                       weights)
    
    #DISPLAY THE RESULT
    print("Average number of iterations before convergence is {}".format(iter_avg))
    print("Probability of getting f!=g is close to {}".format(proba_avg))
    