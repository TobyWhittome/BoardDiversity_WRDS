import pandas as pd
import numpy as np
from scipy.stats import pearsonr

#This adjust the weight for boardsize -- but I don't need it anymore

class mcda:
    
    def __init__(self, predefined_weights):
        # Assume predefined_weights is an array of predefined weights for each column,
        # with the weight for 'boardsize' that will be replaced
        self.weights = predefined_weights
      
    def Normalize(self, dataset, nCol, weights):
        for i in range(1, nCol):
            temp = 0
            for j in range(len(dataset)):
                temp += dataset.iloc[j, i]**2
            temp = temp**0.5
            for j in range(len(dataset)):
                dataset.iat[j, i] = (dataset.iloc[j, i] / temp) * weights[i-1]
        return dataset
        
    def Calc_Values(self, dataset, nCol, impact):
        p_sln = (dataset.max().values)[1:]
        n_sln = (dataset.min().values)[1:]
        for i in range(1, nCol):
            if impact[i-1] == '-':
                p_sln[i-1], n_sln[i-1] = n_sln[i-1], p_sln[i-1]
        return p_sln, n_sln
    
    def calculate_gaussian_weight_for_boardsize(self, dataset):
        mu = dataset['boardsize'].mean()
        print(mu)
        sigma = dataset['boardsize'].std()
        print(sigma)
        # Using the value at the mean; in this context, it's a simplification
        weight = np.exp(-((mu - mu) ** 2) / (2 * sigma ** 2))
        # Normalization not required here as it's a single weight adjustment
        print(weight)
        return weight

    def adjust_weight_for_boardsize(self, dataset):
        # Calculate Gaussian weight for 'boardsize'
        gaussian_weight = self.calculate_gaussian_weight_for_boardsize(dataset)
        # Find the index of 'boardsize' in the columns list
        index_of_boardsize = dataset.columns.get_loc('boardsize') - 1  # Adjusting for zero-based indexing and skipping identifier column
        # Replace the predefined weight for 'boardsize' with the Gaussian weight
        self.weights[index_of_boardsize] = gaussian_weight

# Usage example

if __name__ == "__main__":
    predefined_weights = [0.05, 0.22, 0.17, 0.18, 0.05, 0.07, 0.16, 0.10]  # Example predefined weights
    mcda_instance = mcda(predefined_weights)
    df = pd.read_excel('final_dataset.xlsx')
    
    # Adjust the weight for 'boardsize' before normalization and further processing
    mcda_instance.adjust_weight_for_boardsize(df)
    
    # Proceed with normalization and other calculations as before
