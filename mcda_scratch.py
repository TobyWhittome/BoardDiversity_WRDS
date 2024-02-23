import create_dataset

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
        weights = [0.05, 0.22, 0.17, 0.18, 0.05, 0.07, 0.16, 0.10]
        return weights

    
def main(weights):
    
    inst = mcda()
    df = create_dataset.main()
    no_mcap_df = df.copy()
    no_mcap_df.drop(columns=['mktcapitalisation'], inplace=True)
    
    #columns = [highvotingpower, INED %, 4.5Directors, directorTotalShare%, num_memberships, boardsize, CEO dual, dualclassvotes]
    if len(weights) != 0:
        inst.weights = weights
    else:
        inst.weights = inst.get_weights()
        
    normalized_df = inst.Normalize(no_mcap_df, len(no_mcap_df.columns), inst.weights)

    # Calculating positive and negative values
    impact = ['-', '+', '+', '+', '+', '+', '-', '-']

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
        
    # Appending new columns in dataset   

    #df['distance positive'] = pp
    #df['distance negative'] = nn
    df['Topsis Score'] = score

    # calculating the rank according to topsis score
    df['Rank'] = (df['Topsis Score'].rank(method='max', ascending=False))
    dataset = df.astype({"Rank": int})

    return df
        
if __name__ == "__main__":

  main([])
