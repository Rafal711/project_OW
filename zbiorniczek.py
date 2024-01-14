import numpy as np
import pandas as pd

from copy import deepcopy
from math import inf
from TOPSIS import TOPSIS
from UTA_STAR import UTA_STAR
from FUZZY_TOPSIS import FUZZY_TOPSIS


class slipperyZbiorniczek:
    def __init__(self): 
        self.criterions = None # żeby odwrócić to odjąć -1 i pomnożyć przez -1
        self.criterions_param_list = [1, 1, 1, 0, 0, 0] # dla dataset2: [1, 1, 1, 0, 0, 0], dataset1: [1, 1, 1, 0, 1, 0]
        # self.weights = [1] * len(self.criterions_param_list)
        self.weights = [1, 1, 1, 1, 1, 0] # dla dataset2: [1, 1, 1, 1, 1, 0], dataset1: [1, 1, 1, 1, 0, 1]
        self.loaded_data = pd.DataFrame()
        self.data_to_calculate = pd.DataFrame()
        self.fuzzy_rest = None


    def load_data_from_file(self, path, idx_col):
        self.loaded_data = pd.read_csv(path)
        if path == "dataset3.csv":
            self.data_to_calculate = self.loaded_data.iloc[:, idx_col:-4] # dla dataset2 jest git dla dataset3 [:, 4:-4]
        else:
            self.data_to_calculate = self.loaded_data.iloc[:, idx_col:]
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
        #print(undominated_A0)
        #print(len(undominated_A0))
        self.criterions['Kierunek'] = -(self.criterions['Kierunek'] - 1)
        undominated_A3, domi, not_comparable, compare_counter  = self.naive_owd(data_list)
        #print(undominated_A3)
        #print(len(undominated_A3))
        rest = [el for el in data_list if el not in undominated_A0.tolist()]
        rest = [el for el in rest if el not in undominated_A3.tolist()]
        self.fuzzy_rest = rest
        # rest = [el for el in rest if el not in undominated_A1]
        #print("Rest1: ", rest, len(rest))
        

        self.criterions['Kierunek'] = -(self.criterions['Kierunek'] - 1)
        undominated_A1, domi, not_comparable, compare_counter  = self.naive_owd(rest)
        #print(undominated_A1)
        #print(len(undominated_A1))
        self.criterions['Kierunek'] = -(self.criterions['Kierunek'] - 1)
        undominated_A2, domi, not_comparable, compare_counter = self.naive_owd(rest)
        #print(undominated_A2)
        #print(len(undominated_A2))
        self.criterions['Kierunek'] = -(self.criterions['Kierunek'] - 1)

        rest = [el for el in rest if el not in undominated_A1.tolist()]
        rest = [el for el in rest if el not in undominated_A2.tolist()]

        #print("Rest2: ", rest, len(rest))

        return undominated_A0, undominated_A1, rest
    
    def scoring_do_dataframe_ranking(self, scoring, data = None):
        if np.any(data):
            data["Scoring"] = scoring
            df = data
            #print(df.to_string())
        else:
            self.loaded_data["Scoring"] = scoring
            df = masnyZbiornik.loaded_data
        df = df.groupby(['Nazwa stacji'])["Scoring"].mean()
        df = df.reset_index()
        df.sort_values("Scoring", ascending=False, inplace=True)
        return df

    def run_topsis(self):
        topsis = TOPSIS(self.data_to_calculate, self.criterions_param_list, self.weights)
        topsis_ranking = topsis.run()
        #print(topsis_ranking[0], len(topsis_ranking[0]))
        ranking_list = []
        for el in topsis_ranking:
            ranking_list.append(el[0])
        #print(" ")
        #print(ranking_list)
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
        
    def run_fuzzy_topsis(self):
        data_to_cal = self.undominated_sets()
        fuz_topsis = FUZZY_TOPSIS(data_to_cal, self.criterions_param_list, self.weights, self.fuzzy_rest)
        fuz_topsis_ranking = fuz_topsis.run()
        #print(fuz_topsis_ranking[0], len(fuz_topsis_ranking[0]))
        ranking_list = []
        for el in fuz_topsis_ranking:
            ranking_list.append(el[0])
        #print(" ")
        #print(ranking_list)
        return fuz_topsis_ranking[1], data_to_cal[2]
    
    def fuzzy_output(self, fuzzy_input):
        columnns_to_search = self.data_to_calculate.columns

        matrix_to_search = self.loaded_data[columnns_to_search] # 2d array

        values_to_find = np.array(fuzzy_input[1])

        id = np.argwhere(np.isin(matrix_to_search, values_to_find).all(axis=1)) 
        
        #print("LENGTH", len(id.flatten()))
        print(id)
        #print("LENGTH", len(fuzzy_input[0]))
        if len(id.flatten()) > len(fuzzy_input[0]):
            id = id.flatten()
            id = id[:-1]
        df_= self.loaded_data.iloc[id.flatten()]
        #print("LENGTH", df_.shape)
        #print(df_.to_string())
        df_ = df_.drop(df_.columns[0], axis=1)
        df_ = df_.reset_index()
        #print(df_.to_string())

        df = self.scoring_do_dataframe_ranking(fuzzy_input[0], df_)
        return df
    
    def run_algorithm(self, name, data_path):

        if data_path == "dataset1.csv":
            self.criterions_param_list = [1, 1, 1, 0, 1, 0]
            self.weights = [1, 1, 1, 1, 1, 1]
            self.load_data_from_file(data_path, 4)
        elif data_path == "dataset2.csv":
            self.criterions_param_list = [1, 1, 1, 0, 0, 0]
            self.weights = [1, 1, 1, 1, 1, 0]
            self.load_data_from_file(data_path, 4)
        elif data_path == "dataset3.csv":
            self.criterions_param_list = [1, 1, 1, 0, 1, 0]
            self.weights = [1, 1, 1, 1, 0, 1]
            self.load_data_from_file(data_path, 4)

        if name == "Fuzzy":
           scoring_list = self.run_fuzzy_topsis()
           #print("SCORINGS LENGTH: ", len(scoring_list[0]), len(scoring_list[1]))
           df_out = self.fuzzy_output(scoring_list)
           return df_out
        elif name == "Topsis":
            scoring_list = masnyZbiornik.run_topsis()
            df_out = self.scoring_do_dataframe_ranking(scoring_list)
            return df_out

        else:
            scoring_list = masnyZbiornik.run_uta_star()
            df_out = self.scoring_do_dataframe_ranking(scoring_list)
            return df_out 


if __name__ == "__main__":
    masnyZbiornik = slipperyZbiorniczek()
    df = masnyZbiornik.run_algorithm("Fuzzy", "dataset1.csv")
    print("Output:", df.to_string())
    
