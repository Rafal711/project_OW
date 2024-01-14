import numpy as np
import pandas as pd

from copy import deepcopy
from math import inf
from TOPSIS import TOPSIS


class slipperyZbiorniczek:
    def __init__(self): 
        self.criterions = None # żeby odwrócić to odjąć -1 i pomnożyć przez -1
        self.criterions_param_list = [1, 1, 1, 0, 0, 0]
        self.weights = [1] * len(self.criterions_param_list)
        self.loaded_data = pd.DataFrame()
        self.data_to_calculate = pd.DataFrame()


    def load_data_from_file(self, path):
        self.loaded_data = pd.read_csv(path)
        self.data_to_calculate = self.loaded_data.iloc[:, 4:]
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
        return p, del_elem, not_comparable, compare_counter
    

    def naive_owd_with_optimal_point(self, data_list):
        m = 0
        compare_counter = 0
        P = []
        minArray = []
        noneElements = []
        dominated = []
        X = np.array(data_list, copy=True)
        M = X.shape[0]
        indexes = list(range(0, M))
        dist = lambda minArray, valArray : np.sqrt(np.sum((minArray - valArray)**2, axis = 0))

        if len(X.shape) != 1:
            for col in range(X.shape[1]):
                minArray.append(np.min(X[:, col]))
                # if self.criterions_param_list[col]:
                #     minArray.append(np.max(X[:, col]))
                # else:
                #     minArray.append(np.min(X[:, col]))
        else:
            # if self.criterions[0]: 
            #     minArray.append(np.max(X))
            # else:
            #     minArray.append(np.min(X))
            minArray.append(np.min(X))

        distances = [dist(minArray, X[id]) for id in indexes]

        sortedDistances = [(x, id) for x, id, _ in sorted(zip(distances, indexes, X))]

        while m <= M:
            for i in range(X.shape[0]):

                isNotInNoneElements_1 = True
                isNotInNoneElements_2 = True
                for el in noneElements:
                    if (X[i] == el).all():
                        isNotInNoneElements_1 = False
                    if (X[sortedDistances[m][1]] == el).all():
                        isNotInNoneElements_2 = False

                if (isNotInNoneElements_1) and (isNotInNoneElements_2):
                        compare_counter += 1
                        if self.compare(X[sortedDistances[m][1]], X[i]) and not(np.array_equal(X[sortedDistances[m][1]], X[i])):
                            dominated.append(X[i].tolist())
                            noneElements.append(X[i].tolist())

            if isNotInNoneElements_2:
                P.append(X[sortedDistances[m][1]].tolist())
                noneElements.append(X[sortedDistances[m][1]].tolist())

            M -= 1
            m += 1
        
        for el in X:
            diffrent = 0
            for noneEl in noneElements:
                if (el != noneEl).all():
                    diffrent += 1
            if (diffrent == len(noneElements)):
                P.append(el.tolist())

        return np.array(P), np.array(dominated), compare_counter
    


    # def is_domi(self, p1, p2):
    #     for i in range(len(p1)):
    #         if self.criterions.loc[i].Kierunek == 0:
    #             if p1[i] <= p2[i]:
    #                 continue # compare_tab[i] = True
    #             else:
    #                 return False # compare_tab[i] = False
    #         elif self.criterions.loc[i].Kierunek == 1:
    #             if p1[i] >= p2[i]:
    #                 continue # compare_tab[i] = True
    #             else:
    #                 return False # compare_tab[i] = False
    #     return True


    def is_domi(self, x, y):
        """
        funkcja sprawdzająca czy dany punkt dominuje drugi
        :param x: pkt 1
        :param y: pkt 2
        :return: x,y - większy, None - nieporównywalne, Err - takie same
        """
        len_ = len(x)
        gr = [None] * len_
        eq = [None] * len_
        sm = [None] * len_

        for i in range(len_):
            gr[i] = (x[i] >= y[i])
            eq[i] = (x[i] == y[i])
            sm[i] = (x[i] <= y[i])
        if all(eq):
            raise ValueError(f"the same error {x}")  # dwa takie same punkty
        if all(gr):
            return x  # punkt x większy
        elif all(sm):
            return y  # punkt y większy
        else:
            return None  # punkty nieporównywalne

    def get_rest(self, b, niezdom_dod, niezdom_uj):
        return {key: val for key, val in b.items() if key not in niezdom_dod and key not in niezdom_uj}


    def get_niepor(self, b):
        niepor = dict()
        for pkt in b:
            for pkt2 in b:
                if b[pkt] != b[pkt2]:
                    res = self.is_domi(b[pkt], b[pkt2])
                    if res:
                        break
            else:
                niepor.update({pkt: b[pkt]})
        return niepor

    def str_repr(self, struct):
        for x in struct:
            print(x, struct[x], sep=' ')


    def get_max_min(self, struct):
        min_ = [inf] * 3
        max_ = [-inf] * 3
        for val in struct.values():
            for i in (0, 1, 2):
                min_[i] = min(min_[i], val[i])
                max_[i] = max(max_[i], val[i])
        return {"MIN": min_, "MAX": max_}

    def repr_reslt(self, rest, dod, uj):
        print("NIEZDOM DOD:", len(dod))
        self.str_repr(dod)
        print("NIEZDOM  UJ:", len(uj))
        self.str_repr(uj)
        print("REST:", len(rest))
        self.str_repr(rest)

    def is_gr_gra(self, gra, sma):
        """
        funkcja sprawdzająca czy jeden zbiór dominuje drugi
        :param gra: zbiór potencjalnie większy
        :param sma: zbiór potencjalnie mniejszy
        """
        for g in gra:
            for s in sma:
                res = self.is_domi(gra[g], sma[s])
                if res != gra[g]:
                    raise ValueError(f"Grupy nie są dobrze zdominowane: {g} < {s}")


    def get_niezdom(self, b):
        niezdom_dod = deepcopy(b)
        niezdom_uj = deepcopy(b)
        niepor = self.get_niepor(b)
        print("NIEPOR:", len(niepor))
        self.str_repr(niepor)
        for pkt in b:
            for pkt2 in b:
                if b[pkt] != b[pkt2]:
                    res = self.is_domi(b[pkt], b[pkt2])
                    if res == b[pkt2]:
                        niezdom_dod.pop(pkt, None)
                        niezdom_uj.pop(pkt2, None)
                    if res == b[pkt]:
                        niezdom_dod.pop(pkt2, None)
                        niezdom_uj.pop(pkt, None)
        print("niezdom_dod:", len(niezdom_dod))
        self.str_repr(niezdom_dod)
        print("niezdom_uj:", len(niezdom_uj))
        self.str_repr(niezdom_uj)
        niezdom_dod = self.get_rest(niezdom_dod, niepor, dict())
        niezdom_uj = self.get_rest(niezdom_uj, niepor, dict())
        return niezdom_dod, niezdom_uj, niepor   

    def undominated_sets(self):
        data_list = self.data_to_calculate.to_numpy(copy=True).tolist()
        undominated_A0, domi, comp_count = self.naive_owd_with_optimal_point(data_list)
        print(undominated_A0)
        print(len(undominated_A0))
        self.criterions['Kierunek'] = -(self.criterions['Kierunek'] - 1)
        undominated_A1, domi, comp_count = self.naive_owd_with_optimal_point(data_list)
        print(undominated_A1)
        print(len(undominated_A1))
        rest = [el for el in data_list if el not in undominated_A0.tolist()]
        rest = [el for el in rest if el not in undominated_A1.tolist()]
        # rest = [el for el in rest if el not in undominated_A1]
        print("Rest1: ", rest, len(rest))
        

        self.criterions['Kierunek'] = -(self.criterions['Kierunek'] - 1)
        undominated_A2, domi, comp_count = self.naive_owd_with_optimal_point(rest)
        print(undominated_A2)
        print(len(undominated_A2))
        self.criterions['Kierunek'] = -(self.criterions['Kierunek'] - 1)
        undominated_A3, domi, comp_count = self.naive_owd_with_optimal_point(rest)
        print(undominated_A3)
        print(len(undominated_A3))
        self.criterions['Kierunek'] = -(self.criterions['Kierunek'] - 1)

        rest = [el for el in rest if el not in undominated_A2.tolist()]
        rest = [el for el in rest if el not in undominated_A3.tolist()]

        print("Rest2: ", rest, len(rest))
        rest = np.array(rest)
        for col in range(rest.shape[1]):
            if col < 3:
                print(np.max(rest[:, col]))
            else:
                print(np.min(rest[:, col]))

        # data_dict = deepcopy(data_list)
        # data_name = [f'S{i}' for i in range(len(data_dict))]
        # data = dict(zip(data_name, data_dict))
        # print('\n---ITERACJA 1---\n')
        # print("A0, A3")
        # niezdom_dod, niezdom_uj, nie_por = self.get_niezdom(data)
        # data_rest = self.get_rest(data, niezdom_dod, niezdom_uj)
        # # co z nieporównywalnym
        # self.repr_reslt(data_rest, niezdom_dod, niezdom_uj)
        # glob_max_min = self.get_max_min(data)
        # print(glob_max_min)
        # print('\n---ITERACJA 2---\n')

        # print("A1, A2")
        # niezdom_dod_2, niezdom_uj_2, nie_por_2 = self.get_niezdom(data_rest)
        # rest_2 = self.get_rest(data_rest, niezdom_dod_2, niezdom_uj_2)
        # # co z nieporównywalnym
        # self.repr_reslt(rest_2, niezdom_dod_2, niezdom_uj_2)
        # loc_max_min = self.get_max_min(data_rest)
        # print(loc_max_min)
        # print('\n---KONIEC ITERACJI---\n')

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
        

if __name__ == "__main__":
    masnyZbiornik = slipperyZbiorniczek()
    masnyZbiornik.load_data_from_file('dataset1.csv')
    print(masnyZbiornik.criterions.to_string())
    print(" ")
    ci_list = masnyZbiornik.run_topsis()
    masnyZbiornik.loaded_data["Scoring"] = ci_list
    df = masnyZbiornik.loaded_data
    df = df.groupby(['Nazwa stacji'])["Scoring"].mean()
    df = df.reset_index()
    print(df.to_string())
    df.sort_values("Scoring", ascending=False, inplace=True)
    print("Output:", df.to_string())   
    print(" ")
    print(masnyZbiornik.criterions.to_string())
    # masnyZbiornik.undominated_sets() 
    
