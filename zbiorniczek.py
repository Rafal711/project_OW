import numpy as np
import pandas as pd

from copy import deepcopy
from math import inf
from TOPSIS import TOPSIS
from UTA_STAR import UTA_STAR


class slipperyZbiorniczek:
    def __init__(self): 
        self.criterions = None # żeby odwrócić to odjąć -1 i pomnożyć przez -1
        self.criterions_param_list = [1, 1, 1, 0, 0, 0] # dla dataset2: [1, 1, 1, 0, 0, 0], dataset1: [1, 1, 1, 0, 1, 0]
        self.weights = [1] * len(self.criterions_param_list)
        self.weights = [1, 1, 1, 1, 1, 0] # dla dataset2: [1, 1, 1, 1, 1, 0], dataset1: [1, 1, 1, 1, 0, 1]
        self.loaded_data = pd.DataFrame()
        self.data_to_calculate = pd.DataFrame()


    def load_data_from_file(self, path):
        self.loaded_data = pd.read_csv(path)
        self.data_to_calculate = self.loaded_data.iloc[:, 4:] # dla dataset2 jest git dla dataset3 [:, 4:-1]
        criterions_list = list(self.data_to_calculate.columns)
        self.criterions = pd.DataFrame(list(zip(criterions_list, self.criterions_param_list)), columns=['Nazwa', 'Kierunek'])


    def compare(self, p1, p2):
        for i in range(len(p1)):
            if self.criterions.loc[i].Kierunek == 0:
                if p1[i] <= p2[i]:
                    continue # compare_tab[i] = True
                else:
                    return False # compare_tab[i] = False
            elif self.criterions.loc[i].Kierunek == 1:
                if p1[i] >= p2[i]:
                    continue # compare_tab[i] = True
                else:
                    return False # compare_tab[i] = False
        return True

    def naive_owd(self, data_list):
        X = deepcopy(data_list)
        n = len(X)
        p = []
        del_elem = []
        not_comparable = []
        compare_counter = 0
        for i in range(n):
            Y = X[i]
            fl = 0
            for j in range(i+1, n):
    #             info = f"iter {i + 1}, {j}, {Y}, {X[j]}"
                if Y is None or X[j] is None:
    #                 print(f"skipped iter {i+1}, {j}")
                    continue;
                if (self.compare(Y, X[j])):
                    del_elem.append(X[j])
                    X[j] = None
                elif (self.compare(X[j], Y)):
                    del_elem.append(Y)
                    idx = X.index(Y)
                    X[idx] = None
                    Y = X[j]
                    fl = 1
                else:
                    not_comparable.append(X[j])
    #             info = info + f"{del_elem}, {not_comparable}, {X[i+1:]}"
    #             print(info)
                compare_counter += 1
            if Y is not None and Y not in p:
                p.append(Y)
            if (fl ==0):
                X[i] = None
        # p_nz = list(dict.fromkeys(p))
        # print(p)
        return np.array(p), del_elem, not_comparable, compare_counter

    def undominated_sets(self):
        data_list = self.data_to_calculate.to_numpy(copy=True).tolist()
        undominated_A0, domi, not_comparable, compare_counter = self.naive_owd(data_list)
        print(undominated_A0)
        print(len(undominated_A0))
        self.criterions['Kierunek'] = -(self.criterions['Kierunek'] - 1)
        undominated_A3, domi, not_comparable, compare_counter  = self.naive_owd(data_list)
        print(undominated_A3)
        print(len(undominated_A3))
        rest = [el for el in data_list if el not in undominated_A0.tolist()]
        rest = [el for el in rest if el not in undominated_A3.tolist()]
        # rest = [el for el in rest if el not in undominated_A1]
        print("Rest1: ", rest, len(rest))
        

        self.criterions['Kierunek'] = -(self.criterions['Kierunek'] - 1)
        undominated_A1, domi, not_comparable, compare_counter  = self.naive_owd(rest)
        print(undominated_A1)
        print(len(undominated_A1))
        self.criterions['Kierunek'] = -(self.criterions['Kierunek'] - 1)
        undominated_A2, domi, not_comparable, compare_counter = self.naive_owd(rest)
        print(undominated_A2)
        print(len(undominated_A2))
        self.criterions['Kierunek'] = -(self.criterions['Kierunek'] - 1)

        rest = [el for el in rest if el not in undominated_A1.tolist()]
        rest = [el for el in rest if el not in undominated_A2.tolist()]

        print("Rest2: ", rest, len(rest))

        return undominated_A0, undominated_A1, rest
    
    def scoring_do_dataframe_ranking(self, scoring):
        self.loaded_data["Scoring"] = scoring
        df = masnyZbiornik.loaded_data
        df = df.groupby(['Nazwa stacji'])["Scoring"].mean()
        df = df.reset_index()
        df.sort_values("Scoring", ascending=False, inplace=True)
        return df

    def run_topsis(self):
        topsis = TOPSIS(self.data_to_calculate, self.criterions_param_list, self.weights)
        topsis_ranking = topsis.run()
        print(topsis_ranking[0], len(topsis_ranking[0]))
        ranking_list = []
        for el in topsis_ranking:
            ranking_list.append(el[0])
        print(" ")
        print(ranking_list)
        return topsis_ranking[1]

    def run_uta_star(self):
        intervals = np.random.randint(2, 5, self.data_to_calculate.shape[1])
        weights = np.array(self.weights)

        cryt_name = []
        for i in range(len(self.criterions_param_list)):
            cryt_name.append(f'Kryterium {i+1}')
        cryt_list = deepcopy(self.criterions)
        cryt_list['Nazwa'] = cryt_name

        data_to_cal = deepcopy(self.data_to_calculate)
        data_to_cal.columns = cryt_name

        utaStar = UTA_STAR(cryt_list, data_to_cal, intervals, weights)
        utaRanking = utaStar.run()
        return utaRanking
        

if __name__ == "__main__":
    masnyZbiornik = slipperyZbiorniczek()
    masnyZbiornik.load_data_from_file('dataset1.csv')
    masnyZbiornik.undominated_sets()
    # print(masnyZbiornik.criterions.to_string())
    # # print(masnyZbiornik.data_to_calculate.to_string())
    # print(" ")
    # scoring_list = masnyZbiornik.run_uta_star()
    # scoring_list = masnyZbiornik.run_topsis()
    # df = masnyZbiornik.scoring_do_dataframe_ranking(scoring_list)
    # print("Output:", df.to_string())
    # # masnyZbiornik.undominated_sets() 
    
