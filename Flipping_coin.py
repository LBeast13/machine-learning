# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 15:58:41 2018

@author: Liam
"""

import numpy as np

def coin_flips_analyse(flips,nb_flips):
    """
    Returns the fraction of heads according the flips
    """
    
    count_heads = 0 #counter for the heads
    
    for flip in flips:
        if flip == 1:
            count_heads = count_heads + 1
            
    #fraction of heads for the sample
    fraction = count_heads/nb_flips
    
    return fraction


def flip_coins(nb_coins,nb_flips):
    """
    Flips nb_coins coins with nb_flips flips each.
    
    Returns a 2D array (each line is a coin, each column is a flip except 
    the last which is the fractionof heads) and the minimum fraction of heads
    obtained by a coin of our sample.
    """
    coins_data = [] #all our coins with their corresponding flips and frequencies
    minimum_heads = 1 #to store the minimum fraction of heads obtained by a coin in an experience

    #Loop for each coin
    for i in range(nb_coins):
        
        #Flips the coin (1=heads)
        flips = np.random.binomial(1,0.5,size=nb_flips)
        
        #Get the fraction of heads for the flips
        fraction = coin_flips_analyse(flips,nb_flips)
        flips = np.append(flips,fraction)
        
        coins_data.append(flips)
       
        #Update the minimum fraction 
        if minimum_heads > fraction:
            minimum_heads = fraction
        
    return coins_data, minimum_heads


if __name__ == '__main__':
    
    #The number of coins flipped in one experience
    nb_coins = 1000
    
    #The number of flips done by a single coin
    nb_flips = 10
    
    #Number of independent experiences done
    nb_experiences = 100000
     
    sum_min_freq = 0
    sum_rand_freq = 0
    sum_first_freq = 0
    
    head_frequencies = []
    curr_experience = []
    
    for i in range(nb_experiences):
        
        #GET the min fraction, the random fraction and the first fraction
        coins_data, min_freq = flip_coins(nb_coins,nb_flips)
        first_freq = coins_data[0][10]
        rand_freq = coins_data[np.random.randint(1000)][10]
        
        curr_experience.append(first_freq)
        curr_experience.append(rand_freq)
        curr_experience.append(min_freq)
        
        head_frequencies.append(curr_experience)
        
        sum_min_freq = sum_min_freq + min_freq
        sum_rand_freq = sum_rand_freq + rand_freq
        sum_first_freq = sum_first_freq + first_freq
        
        #Display every 1000 experiences done
        if (i+1)%1000 == 0:
            print("Step {}".format(i+1))
    
    #Fraction AVERAGES (min,random,first)    
    min_freq_average = sum_min_freq/nb_experiences
    rand_freq_average = sum_rand_freq/nb_experiences
    first_freq_average = sum_first_freq/nb_experiences
    
    print("The average value of the minimum frequency is {}".format(min_freq_average))
    print("The average value of the first coin frequency is {}".format(first_freq_average))
    print("The average value of the random coin frequency is {}".format(rand_freq_average))    
    