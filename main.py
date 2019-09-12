#content based recommendation because we have feature matrix of users and movies.

#collaborative filtering algorithm.  both user and movie collaborate to improve themselves . users help system learn its featur matrix and
#in return system is helping user to learn its feature matrix.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from scipy.optimize import minimize

#y is a matrix of shape 1682 X 943   1642 movies and 943 users
#R is of same shape indicating rated or unrated movies


data = loadmat('ex8_movies.mat')
Y=data['Y']
R=data['R']

'''
plt.imshow(Y)
plt.show()
'''
#cost function of collaborative filtering algorithm

def cost(parameters,Y,R,features,lemda):
    Y=np.matrix(Y)
    R=np.matrix(R)

    movies=Y.shape[0]
    users=Y.shape[1]

    feature_matrix=np.matrix(np.reshape(parameters[:movies*features],(movies,features)))
    theta=np.matrix(np.reshape(parameters[movies*features:],(users,features)))

    cost=0
    error=np.multiply(feature_matrix.dot(theta.T)-Y,R)
    squared_error=np.power(error,2)
    cost=(np.sum(squared_error))/2
    
    #regularisation added.
    cost+=(lemda*(np.sum(np.square(theta))))/2
    cost+=(lemda*(np.sum(np.square(feature_matrix))))/2

    grads=gradients(error,feature_matrix,theta,lemda)

    return cost,grads

#gradients calculation

def gradients(error,feature_matrix,theta,lemda):
    feature_matrix_grad=np.zeros(feature_matrix.shape)
    theta_grad=np.zeros(theta.shape)
    
    #regularization factor included to avoid over fitting.
    feature_matrix_grad=error.dot(theta)+lemda*feature_matrix
    theta_grad=(error.T).dot(feature_matrix)+lemda*theta
    
    #optimiser function needs flattened parameters.
    grads=np.concatenate((np.ravel(feature_matrix_grad),np.ravel(theta_grad)))
    return grads

#movies ids with their names

movie_idx = {}
f = open('movie_ids.txt')
for line in f:
   tokens = line.split(' ')
   tokens[-1] = tokens[-1][:-1]
   #movie_idx stores index of movies along with movies name.
   movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])

ratings = np.zeros((1682, 1))

#rating by a new user.user rates 10 movies of his choice. he may rate any number of movies including zero.
ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5

print('Rated {0} with {1} stars.'.format(movie_idx[0], str(int(ratings[0]))))
print('Rated {0} with {1} stars.'.format(movie_idx[6], str(int(ratings[6]))))
print('Rated {0} with {1} stars.'.format(movie_idx[11], str(int(ratings[11]))))
print('Rated {0} with {1} stars.'.format(movie_idx[53], str(int(ratings[53]))))
print('Rated {0} with {1} stars.'.format(movie_idx[63], str(int(ratings[63]))))
print('Rated {0} with {1} stars.'.format(movie_idx[65], str(int(ratings[65]))))
print('Rated {0} with {1} stars.'.format(movie_idx[68], str(int(ratings[68]))))
print('Rated {0} with {1} stars.'.format(movie_idx[97], str(int(ratings[97]))))
print('Rated {0} with {1} stars.'.format(movie_idx[182], str(int(ratings[182]))))
print('Rated {0} with {1} stars.'.format(movie_idx[225], str(int(ratings[225]))))
print('Rated {0} with {1} stars.'.format(movie_idx[354], str(int(ratings[354]))))

#adding new user to model
Y=np.append(Y,ratings,axis=1)
R=np.append(R,ratings!=0,axis=1)


movies=Y.shape[0]
users=Y.shape[1]
features=10
alpha=10

features_matrix= np.random.random(size=(movies, features))
theta = np.random.random(size=(users, features))

#parameters needs to be flattened before feeding it to a minimise optiser function.
parameters = np.concatenate((np.ravel(features_matrix), np.ravel(theta)))

#new normalised matrix.
Y_normalised=np.zeros((movies,users))
Y_mean=np.zeros((movies,1))

#mean normalisation . A technique useful for new user
#parameter matrix of new user will become zero intitially if user hasn't rated any movie initially.
#so for each movie prediction will be zero.  mean normalisation handles this case efficiently as to finally adds mean of movies to the users
#to the predicted ratings.


for i in range(movies):
    cnt=0
    cos=0
    for j in range(users):
        if  R[i][j]==1:
            cnt+=1
            cos+=Y[i][j]
    for j in range(users):
        if R[i][j]==1:
           Y_normalised[i][j]=Y[i][j]-cos/cnt
    Y_mean[i]=cos/cnt

#optimisation function to optimize a multivariable scalar function. 
#func takes optimisation function.
#x0 takes parameters to optimize.
#args takes remaining parameters of the optimisation function apart from parameters to be optimised.
fmin = minimize(fun=cost, x0=parameters, args=(Y_normalised, R, features,alpha), method='CG', jac=True, options={'maxiter': 100})

print(fmin)

#fmin returns our optimised parameters.
X = np.matrix(np.reshape(fmin.x[:movies * features], (movies, features)))
Theta = np.matrix(np.reshape(fmin.x[movies * features:], (users, features)))

print(X.shape)
print(theta.shape)

#our new prediction matrix after adding of new user.
predictions = X * Theta.T

#adding mean value that we subtracted earlier during mean normalisation.
my_preds = predictions[:, -1] + Y_mean

#returns index of elements sotred by ratings in decreasing order.
idx = np.argsort(my_preds, axis=0)[::-1]

#prints top 10 rated movies rated by users  uses our earlier made movie array.
print("Top 10 movie predictions:")
for i in range(10):
    j = int(idx[i])
    print('Predicted rating of {0} for movie {1}.'.format(str(float(my_preds[j])), movie_idx[j]))
