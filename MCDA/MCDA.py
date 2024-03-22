import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy import stats

class mcda:
    
    def __init__(self):
        self.weights = []
      
    def Normalize(self, dataset, nCol, weights):
        for i in range(1, nCol):
            dataset.iloc[:, i] = dataset.iloc[:, i].astype(float)
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
    
        #Normalised indiv spearman's coeff scores
        #weights = [0.00335242, 0.34013028, 0.11792744, 0.19255043, 0.10105673, 0.12141488, 0.02420027, 0.09936755]
        
        
        #Trial and error ---- 1919
        weights = [0.303, 0.04, 0.277, 0.032, 0.101, 0.121, 0.024, 0.099]
        

        #Trial and error with new data ---- -0.4525
        weights = [0.15, 0.132, 0.203, 0.04, 0.177, 0.101, 0.151, 0.024, 0.019]
        
        return weights

    
def main(df, weightsin):
    
    inst = mcda()

    df = pd.read_excel('dataset/transformed_dataset.xlsx')
    df.dropna(inplace=True)
    no_mcap_df = df.copy()
    no_mcap_df.drop(columns=['ticker', 'tobinsQ'], inplace=True)

    
    if len(weightsin) != 0:
        inst.weights = weightsin
    else:
        inst.weights = inst.get_weights()

    normalized_df = inst.Normalize(no_mcap_df, len(no_mcap_df.columns), inst.weights)

    impact = ['-', '+', '+', '-', '+', '+', '-', '-', '-']

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
        

    #df['Topsis Score'] = score
    #df['Rank'] = (df['Topsis Score'].rank(method='max', ascending=False))
    #dataset = df.astype({"Rank": int})

    """ 
    plt.figure(figsize=(10, 6)) # Optional: Adjusts the figure size
    plt.scatter(df['Topsis Score'], df['tobinsQ'], color='b') # You can customize the plot with markers, linestyles, and colors
    plt.title('Plot of A vs B') # Optional: Adds a title to the plot
    plt.xlabel('Topsis Score') # Label for the X-axis
    plt.ylabel('TobinsQ') # Label for the Y-axis
    plt.grid(True) # Optional: Adds a grid for easier visualization
    #plt.show() """
    
    correlationspear, _ = stats.spearmanr(score, df['tobinsQ'])
    #correlationspear, _ = stats.spearmanr(df['Topsis Score'], df['tobinsQ'])
    #print(f"Spearman's rank correlation: {correlationspear} \n")
    
    return correlationspear
        
if __name__ == "__main__":
    df = pd.read_excel('dataset/transformed_dataset.xlsx')
    df.dropna(inplace=True)
    
    main(df, [])

