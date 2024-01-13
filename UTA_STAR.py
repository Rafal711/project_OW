import numpy as np

from operator import itemgetter
from beautifultable import BeautifulTable

class UTA_STAR:
    def __init__(self, criterions, generated_data, intervals, weights) -> None:
        self.criterions = criterions # pd.DataFrame(columns=['Nazwa', 'Kierunek'])
        self.generated_data = generated_data # pd.DataFrame(columns=['Kryterium 1', 'Kryterium 2'])
        self.intervals_per_criterion = np.array(intervals) # np.array([1, 1])
        self.weights = np.array(weights) # np.array([1, 1])

    def determineIdealPoints(self):
        directionToMinOrMax = {
            "Min": np.min,
            "Max": np.max
        }
        num_of_criterions = self.generated_data.shape[1]
        return [directionToMinOrMax[self.criterions.loc[i].Kierunek](self.generated_data[f"Kryterium {i}"].to_numpy()) for i in range(num_of_criterions)]
    
    def determineAntiIdealPoints(self):
        directionToAMinOrAMax = {
            "Min": np.max,
            "Max": np.min
        }
        num_of_criterions = self.generated_data.shape[1]
        return [directionToAMinOrAMax[self.criterions.loc[i].Kierunek](self.generated_data[f"Kryterium {i}"].to_numpy()) for i in range(num_of_criterions)]

    def determine_intermediate_points(self):
        intermediate_points_mat = []
        min_per_criterion = self.generated_data.min().to_numpy()
        max_per_criterion = self.generated_data.max().to_numpy()
        part_equation = (max_per_criterion - min_per_criterion) / (self.intervals_per_criterion - 1)
        for i in range(self.generated_data.shape[1]):
            temp = []
            for j in range(self.intervals_per_criterion[i]):
                if self.criterions.loc[i].Kierunek == "Min":
                    temp.append(min_per_criterion[i] + part_equation[i] * j)
                else:
                    temp.append(max_per_criterion[i] - part_equation[i] * j)
            intermediate_points_mat.append(temp)
        return intermediate_points_mat
    
    def calculate_utility_function_values(self):
        utility_function_values_mat = []
        norm_weights = (1 / self.weights.sum()) * self.weights
        part_equation = norm_weights / (self.intervals_per_criterion - 1)
        for i in range(self.weights.shape[0]):
            temp = []
            for j in range(self.intervals_per_criterion[i]):
                temp.append((norm_weights[i] - (j * part_equation[i])).tolist())
            utility_function_values_mat.append(temp)
        return utility_function_values_mat
    
    def determine_gradient_and_intercept_for_utility_functions(self, intermediate_points_mat, utility_funs_values_mat):
        coefficients = []
        for i in range(len(intermediate_points_mat)):
            intermediate_points, utility_fun_vals = intermediate_points_mat[i], utility_funs_values_mat[i]
            temp = []
            for j in range(1, len(intermediate_points)):
                a = (utility_fun_vals[0] - utility_fun_vals[j]) / (intermediate_points[0] - intermediate_points[j])
                b = utility_fun_vals[0] - a * intermediate_points[0]
                temp.append((a, b))
            coefficients.append(temp)
        return coefficients
    
    def calculate_scoring(self, intermediate_points_mat, utility_funs_coeff_mat):
        all_scorings = []
        for i in range(self.generated_data.shape[1]):
            scores = []
            alternative_value_for_criterion_i = self.generated_data[f'Kryterium {i+1}'].values
            utility_fun_coeffs = utility_funs_coeff_mat[i]
            for value in alternative_value_for_criterion_i:
                for j, intermediate_point in enumerate(intermediate_points_mat[i]):
                    if self.criterions.loc[i].Kierunek == "Min" and value >= intermediate_point:
                        a, b = utility_fun_coeffs[j - 1]
                        scores.append(a * value + b)
                        break
                    elif self.criterions.loc[i].Kierunek == "Max" and value <= intermediate_point:
                        a, b = utility_fun_coeffs[j - 1]
                        scores.append(a * value + b)
                        break
            all_scorings.append(scores)
        return np.array(all_scorings)
    
    def create_ranking(self, all_scorings):
        raking_per_criterion = list(zip(np.arange(1, df_data.shape[0] + 1), np.sum(all_scorings.T, axis=1)))
        return sorted(raking_per_criterion, key=itemgetter(1), reverse=True)

    def run(self):
        intermediate_points_mat = self.determine_intermediate_points()
        utility_function_values_mat = self.calculate_utility_function_values()
        utility_funs_coeff_mat = self.determine_gradient_and_intercept_for_utility_functions(intermediate_points_mat, utility_function_values_mat)
        all_scorings = self.calculate_scoring(intermediate_points_mat, utility_funs_coeff_mat)
        rankings = self.create_ranking(all_scorings)

        table = BeautifulTable()
        table.columns.header = ["ranking", "alternatywa", "scoring"]

        for i in range(len(rankings)):
            table.rows.append([i + 1, rankings[i][0], rankings[i][1]])
        print(table)
        return rankings