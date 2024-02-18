from scikitmcda.topsis import TOPSIS
from scikitmcda.wsm import WSM
from scikitmcda.wpm import WPM
from scikitmcda.waspas import WASPAS
from scikitmcda.promethee_ii import PROMETHEE_II
from scikitmcda.electre_i import ELECTRE_I
#from scikitmcda.electre_i import ELECTRE_II
from scikitmcda.vikor import VIKOR
from scikitmcda.constants import MAX, MIN, LinearMinMax_, LinearMax_, LinearSum_, Vector_, EnhancedAccuracy_, Logarithmic_ 
import create_dataset
import pandas as pd


topsis = TOPSIS()

df = create_dataset.main()
indexes = df[df.columns[0]]
del df[df.columns[0]]
topsis.dataframe(df.values, indexes, df.columns)
print(topsis.pretty_original())


topsis.set_weights_manually([0.5918, 0.2394, 0.1151, 0.0537, 0.2, 0.1])
# topsis.set_weights_by_entropy()
# topsis.set_weights_by_ranking_B_POW(0)

                                   # C1   C2     C3   C4 
""" w_AHP = topsis.set_weights_by_AHP([[  1,    4,    5,   7],   # C1
                                   [1/4,    1,    3,   5],   # C2
                                   [1/5,  1/3,    1,   3],   # C3
                                   [1/7,  1/5,  1/3,   1]])  # C4
print("AHP Returned:\n", w_AHP) """

topsis.set_signals([MIN, MAX, MAX, MAX, MIN, MIN])

topsis.decide()

print("WEIGHTS:\n", topsis.weights)

print("NORMALIZED:\n", topsis.pretty_normalized())

print("WEIGHTED:\n", topsis.pretty_weighted())

print("RANKING TOPSIS with", topsis.normalization_method , ":\n", topsis.pretty_decision())

topsis.decide(EnhancedAccuracy_)
print("RANKING TOPSIS with", topsis.normalization_method, ":\n", topsis.pretty_decision())
