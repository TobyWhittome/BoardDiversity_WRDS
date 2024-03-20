import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy import stats

class mcda:
    
    def __init__(self):
        self.weights = []
      
    def Normalize(self, dataset, nCol, weights):
        for i in range(1, nCol):
            temp = 0
            # Calculating Root of Sum of squares of a particular column
            for j in range(len(dataset)):
                temp = temp + dataset.iloc[j, i]**2
            temp = temp**0.5
            # Weighted Normalizing a element
            for j in range(len(dataset)):
                dataset.iat[j, i] = (dataset.iloc[j, i] / temp)*weights[i-1]
        return dataset
        
        
    # Calculate ideal best and ideal worst
    def Calc_Values(self, dataset, nCol, impact):
        p_sln = (dataset.max().values)[1:]
        n_sln = (dataset.min().values)[1:]
        for i in range(1, nCol):
            if impact[i-1] == '-':
                p_sln[i-1], n_sln[i-1] = n_sln[i-1], p_sln[i-1]
        return p_sln, n_sln
    
    def get_weights(self):
        #Weights OG
        #High voting power, INEDS, Num directors, total_share, num_commttiees, CEOdual, dualclass, boardsize
        #weights = [0.05, 0.22, 0.17, 0.18, 0.05, 0.07, 0.16, 0.10]
        
        #Normalised indiv spearman's coeff scores
        #weights = [0.00335242, 0.34013028, 0.11792744, 0.19255043, 0.10105673, 0.12141488, 0.02420027, 0.09936755]
        
        #Trial and error ---- 1.833
        weights = [0.303, 0.04, 0.117, 0.192, 0.101, 0.121, 0.024, 0.099]
        
        
        #Trial and error ---- 1.833
        weights = [0.303, 0.04, 0.117, 0.192, 0.101, 0.121, 0.024, 0.099]
        
        #Trial and error ---- 1.888
        weights = [0.303, 0.04, 0.117, 0.092, 0.101, 0.121, 0.024, 0.099]
        
        
        #Trial and error ---- 1919
        weights = [0.303, 0.04, 0.277, 0.032, 0.101, 0.121, 0.024, 0.099]
        
        
        #Trial and error ---- 
        weights = [0.303, 0.04, 0.277, 0.032, 0.101, 0.121, 0.024, 0.099]
        
        
        #Trial and error ---- 1.833
        #weights = [0.303, 0.04, 0.117, 0.192, 0.101, 0.121, 0.024, 0.099]
        
        #Pearson optimize weights with Tobin'Q -- 0.12 val
        #weights = [0.00000000e+00, 1.03180555e-16, 8.67909704e-01, 1.32090296e-01, 2.85622294e-18, 1.48625612e-18, 1.66028900e-16, 0.00000000e+00]
        
        # PearsonOp with OG as initial guess -- 0.125 val
        #weights = [3.67845656e-17, 9.07604978e-17, 8.64099889e-01, 1.35900111e-01, 0, 0, 3.55785703e-17, 2.56436079e-17]
        
        # Entropy weights? -0.123 val
        #weights = [0.2740107476113237, 6.437404949557777e-05, 0.10460841357171742, 0.0724586189525137, 0.0001823365247053887, 0.0006540140675967686, 0.2740107476113237, 0.2740107476113237]
        
       
        return weights

    
def main(weights):
    
    inst = mcda()
    df = pd.read_excel('dataset/transformed_dataset.xlsx')
    no_mcap_df = df.copy()
    no_mcap_df.drop(columns=['ticker','mktcapitalisation', 'tobinsQ'], inplace=True)
    
    #columns = [highvotingpower, INED %, 4.5Directors, directorTotalShare%, num_memberships, boardsize, CEO dual, dualclassvotes]
    if len(weights) != 0:
        inst.weights = weights
    else:
        inst.weights = inst.get_weights()
        
    normalized_df = inst.Normalize(no_mcap_df, len(no_mcap_df.columns), inst.weights)

    # Calculating positive and negative values
    #impact = ['-', '+', '+', '+', '+', '-', '-', '-']
    
    #According to my regressions
    impact = ['+', '-', '+', '+', '-', '-', '-', '-']

    p_sln, n_sln = inst.Calc_Values(normalized_df, len(normalized_df.columns), impact)

    # calculating topsis score
    score = [] # Topsis score
    pp = [] # distance positive
    nn = [] # distance negative
    
    
    # Calculating distances and Topsis score for each row
    for i in range(len(normalized_df)):
        temp_p, temp_n = 0, 0
        for j in range(1, len(normalized_df.columns)):
            temp_p = temp_p + (p_sln[j-1] - normalized_df.iloc[i, j])**2
            temp_n = temp_n + (n_sln[j-1] - normalized_df.iloc[i, j])**2
        temp_p, temp_n = temp_p**0.5, temp_n**0.5
        score.append(temp_n/(temp_p + temp_n))
        nn.append(temp_n)
        pp.append(temp_p)
        

    df['Topsis Score'] = score
    df['Rank'] = (df['Topsis Score'].rank(method='max', ascending=False))
    dataset = df.astype({"Rank": int})

    plt.figure(figsize=(10, 6)) # Optional: Adjusts the figure size
    plt.scatter(df['Topsis Score'], df['tobinsQ'], color='b') # You can customize the plot with markers, linestyles, and colors
    plt.title('Plot of A vs B') # Optional: Adds a title to the plot
    plt.xlabel('Topsis Score') # Label for the X-axis
    plt.ylabel('TobinsQ') # Label for the Y-axis
    plt.grid(True) # Optional: Adds a grid for easier visualization
    #plt.show()
    
    correlationspear, _ = stats.spearmanr(df['Topsis Score'], df['tobinsQ'])
    print(f"Spearman's rank correlation: {correlationspear} \n")
    
    return df
        
if __name__ == "__main__":

  main([])
