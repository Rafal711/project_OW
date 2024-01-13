import numpy as np
import pandas as pd


class slipperyZbiorniczek:
    def __init__(self): 
        self.criterions = pd.DataFrame(columns=['Nazwa', 'Kierunek']) # żeby odwrócić to odjąć -1 i pomnożyć przez -1
        self.loaded_data = pd.DataFrame()

    def load_data_from_file(self, path):
        df = pd.read_csv(path)
        self.loaded_data = df

    def compare(self, p1, p2):
        for i in range(len(p1)):
            if self.criterions.loc[i].Kierunek == "Min":
                if p1[i] <= p2[i]:
                    continue # compare_tab[i] = True
                else:
                    return False # compare_tab[i] = False
            elif self.criterions.loc[i].Kierunek == "Max":
                if p1[i] >= p2[i]:
                    continue # compare_tab[i] = True
                else:
                    return False # compare_tab[i] = False
        return True

    def naive_owd(self):
        X = self.loaded_data.to_numpy(copy=True).tolist()
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
        return p, del_elem, not_comparable, compare_counter
    
