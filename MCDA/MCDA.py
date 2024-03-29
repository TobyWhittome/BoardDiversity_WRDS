import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

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
        
        #0.1422
        weights = [2, 30, 10, 13, 3, 3, 4, 16, 10]
        return weights

    
def main(df, weightsin):
    
    inst = mcda()

    #df = pd.read_excel('dataset/transformed_dataset.xlsx')
    #df.dropna(inplace=True)
    no_mcap_df = df.copy()
    no_mcap_df.drop(columns=['ticker', 'tobinsQ'], inplace=True)

    
    if len(weightsin) != 0:
        inst.weights = weightsin
    else:
        inst.weights = inst.get_weights()

    #Steps 1, 2 and 3
    normalized_df = inst.Normalize(no_mcap_df, len(no_mcap_df.columns), inst.weights)

    impact = ['-', '+', '+', '-', '+', '+', '-', '-', '-']

    #Steps 4
    p_sln, n_sln = inst.Calc_Values(normalized_df, len(normalized_df.columns), impact)

    # calculating topsis score
    score = [] # Topsis score
    pp = [] # distance positive
    nn = [] # distance negative
    
    
    # Calculating distances and Topsis score for each row
    #Step 5
    for i in range(len(normalized_df)):
        temp_p, temp_n = 0, 0
        for j in range(1, len(normalized_df.columns)):
            temp_p = temp_p + (p_sln[j-1] - normalized_df.iloc[i, j])**2
            temp_n = temp_n + (n_sln[j-1] - normalized_df.iloc[i, j])**2
        temp_p, temp_n = temp_p**0.5, temp_n**0.5
        #Step 6
        score.append(temp_n/(temp_p + temp_n))
        nn.append(temp_n)
        pp.append(temp_p)
        

    #df['Topsis Score'] = score
    #df['Rank'] = (df['Topsis Score'].rank(method='max', ascending=False))
    #sort in order of rank
    #df = df.sort_values(by='Rank')
    #dataset = df.astype({"Rank": int})


    #Graph for Topsis Score vs Tobin's Q
    """sns.set_theme()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Topsis Score', y='tobinsQ', data=df, color='b')
    plt.title('Scatter plot of MCDA TOPSIS Score vs Tobin\'s Q for each company')
    plt.xlabel('Topsis Score')
    plt.ylabel('Tobins Q')
    plt.ylim(-0.5, 15)
    plt.grid(True)
    plt.show() """
    
    correlationspear, _ = stats.spearmanr(score, df['tobinsQ'])
    #correlationspear, _ = stats.spearmanr(df['Topsis Score'], df['tobinsQ'])
    print(f"Spearman's rank correlation: {correlationspear} \n")

    #create new dataframe
    output = pd.DataFrame(columns=['ticker', 'Topsis Score'])
    output['Topsis Score'] = score
    output['Rank'] = (output['Topsis Score'].rank(method='max', ascending=False))
    output['ticker'] = df['ticker']
    output['tobinsQ'] = df['tobinsQ']
    output = output.sort_values(by='Rank')

    #output excel file of output
    #output.to_excel('dataset/MCDATobinsQ.xlsx', index=False)
    
    return output, correlationspear
        
if __name__ == "__main__":
    df = pd.read_excel('dataset/transformed_dataset.xlsx')
    df.dropna(inplace=True)
    
    main(df, [])

