import numpy as np
from numpy.linalg import norm 
import pandas as pd 
import math
import plotly as plt 

def cos_sim(a,b):
	cos_sim = abs(np.dot(a,b))/(norm(a)*norm(b))
	return(cos_sim)


colummn_header = ['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10', 'D11']
query = np.array([0.1, 0.1, 0, 0, 0, 0.5, 0, 0.35, 0.3, 0])
doc_term = np.array([
    #D1   D2   D3   D4   D5   D6   D7   D8   D9   D10  D11
    [0.1, 0.0, 0.1, 0.2, 0.0, 0.1, 0.9, 0.9, 0.3, 0.0, 0.8],
    [0.1, 0.0, 0.1, 0.0, 0.0, 0.1, 0.9, 0.0, 0.3, 0.0, 0.8],
    [0.0, 0.9, 0.0, 0.3, 0.1, 0.7, 0.0, 0.2, 0.7, 0.5, 0.5],
    [0.0, 0.9, 0.1, 0.0, 0.1, 0.9, 0.3, 0.8, 0.4, 0.1, 0.4],
    [0.0, 0.0, 0.0, 0.5, 0.9, 0.3, 0.7, 0.4, 0.6, 0.0, 0.3],
    [0.6, 0.0, 0.7, 0.3, 0.3, 0.9, 0.1, 0.0, 0.0, 0.0, 0.3],
    [0.0, 0.8, 0.0, 0.6, 0.6, 0.0, 0.1, 0.4, 0.9, 0.3, 0.1],
    [0.4, 0.0, 0.5, 0.5, 0.1, 0.7, 0.1, 0.5, 0.3, 0.8, 0.1],
    [0.3, 0.0, 0.2, 0.8, 0.7, 0.7, 0.8, 0.0, 0.6, 0.8, 0.0],
    [0.0, 0.5, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0, 0.4, 0.5, 0.3]
])

cos_sim_matrix = np.empty((11,11))
cos_dist_matrix = np.empty((11,11))
cos_angle_matrix = np.empty((11,11))

for i in range(11):
	for j in range(11):
		cos_sim_matrix[i,j] = round(cos_sim(doc_term[:,i],doc_term[:,j]),2)
		cos_dist_matrix[i,j] = 1 - cos_sim_matrix[i,j]
		cos_angle_matrix[i,j] = round(np.arccos(cos_sim_matrix[i,j])/(22/7)*180,1)


cos_sim_matrix = pd.DataFrame(cos_sim_matrix)
cos_sim_matrix.index = colummn_header
cos_sim_matrix.columns = colummn_header
# print('Cosine Similarity Matrix')
# print(cos_sim_matrix)
# print('=====================================================')

cos_dist_matrix = pd.DataFrame(cos_dist_matrix)
cos_dist_matrix.index = colummn_header
cos_dist_matrix.columns = colummn_header
# print(' ')
# print('Cosine Distance Matrix')
# print(cos_dist_matrix)
# print('======================================================')

cos_angle_matrix = pd.DataFrame(cos_angle_matrix)
cos_angle_matrix.index = colummn_header
cos_angle_matrix.columns = colummn_header
# print(' ')
# print('Cosine Angle Matrix')
# print(cos_angle_matrix)


# Query ranking
query_cos_sim = np.empty(11)

for i in range(11):
	query_cos_sim[i] = round(cos_sim(query, doc_term[:,i]),2)

query_cos_sim = pd.DataFrame(query_cos_sim)
query_cos_sim.index = colummn_header
query_cos_sim.rename(columns = {0:'cos_sim'}, inplace = True)
query_cos_sim = query_cos_sim.sort_values('cos_sim', ascending = False)
query_cos_sim.loc[:,'rank'] = range(1,12)
# print(query_cos_sim)



