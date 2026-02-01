import numpy as np
import pandas as pd
import pickle

import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

class ML:
    def __init__(self):
        self.N_A = 6.022 * 10 ** 23  # molecules/mol
        self.A_cs = (
            0.142 * 10 ** -18
        )  # m2/molecule, for argon.

        self.norm_vals = np.array(
            [
                [
                    1.58953000e-04, 1.00155915e-02, 4.39987176e-02, 3.24635588e-01, 2.05368726e00,
                    2.71670605e00, 2.81671833e00, 2.52660562e-08, 1.59200831e-06, 6.99372815e-06,
                    5.16018006e-05, 6.20679622e-04, 2.02395972e-03, 2.28557690e-03, 1.00312072e-04,
                    4.40673180e-04, 3.25141742e-03, 3.91088782e-02, 1.27529230e-01, 1.44013667e-01,
                    1.93588715e-03, 1.42835495e-02, 1.71806178e-01, 5.60238764e-01, 6.32655265e-01,
                    1.05388265e-01, 1.26763694e00, 4.13360776e00, 4.66791819e00, 4.21763134e00,
                    6.85987932e00, 7.11241758e00, 7.38049176e00, 7.65219574e00, 7.93390218e00,
                ],
                [
                    4.78708045e00, 2.68703345e01, 3.80141609e01, 4.78106234e01, 6.17790024e01,
                    1.06410565e02, 1.61478833e02, 2.29161393e01, 1.28630453e02, 1.50296581e02,
                    1.60737108e02, 1.66083055e02, 1.72479679e02, 1.80115011e02, 7.22014876e02,
                    8.43628897e02, 9.02232560e02, 9.32239866e02, 9.68144720e02, 1.01100256e03,
                    1.44507643e03, 1.65793236e03, 1.73340193e03, 1.82180829e03, 1.87702758e03,
                    2.28585571e03, 2.95369262e03, 3.20344500e03, 3.25911476e03, 3.81664514e03,
                    4.13936531e03, 5.18387215e03, 1.13232083e04, 1.71830538e04, 2.60754134e04,
                ],
            ]
        )

    def initialize_multiple_cols(self, data, col_list, default_val=np.nan):
        for col in col_list:
            data[col] = default_val
        return data

    def pressure_bin_features(self, data, pressure_bins, isotherm_df, pressure_col, loading_col):
        feature_list = ["c_%d" % i for i in range(len(pressure_bins))]
        data = self.initialize_multiple_cols(data, feature_list)

        isotherm_df.loc[isotherm_df.shape[0]] = [0.0, 0.0]

        for i, _ in enumerate(pressure_bins):
            try:
                val = isotherm_df.loc[
                    (isotherm_df[pressure_col] >= pressure_bins[i][0]) &
                    (isotherm_df[pressure_col] < pressure_bins[i][1]), loading_col
                ].mean()
                data["c_%d" % i] = val
            except Exception:
                print("Feature issue in pressure_bin_features")

        return data

    def normalize_df(self, df, col_list):
        df[col_list] = (df[col_list] - self.norm_vals[0, :]) / (
            self.norm_vals[1, :] - self.norm_vals[0, :]
        )
        return df

    def generate_combine_feature_list(self, feature_list):
        out_feature_list = []
        for n1 in np.arange(0, len(feature_list)):
            for n2 in np.arange(n1, len((feature_list))):
                el1 = feature_list[n1]
                el2 = feature_list[n2]
                out_feature_list.append(el1 + "-" + el2)
        return out_feature_list

    def combine_features(self, data, feature_list):
        out_feature_list = self.generate_combine_feature_list(feature_list)
        for feature in out_feature_list:
            data[feature] = data[feature.split("-")[0]] * data[feature.split("-")[1]]
        return data

    def build_pressure_bins(self, init_val, final_val, n_points):
        p_points = np.logspace(
            np.log10(init_val), np.log10(final_val), n_points
        )
        p_points = np.insert(p_points, 0, 0)

        p_points = np.round(p_points, decimals=0)

        bins = [(p_points[i], p_points[i + 1]) for i in range(len(p_points) - 1)]
        return bins

    def add_features(
        self, tr_data, pressure_bins, feature_list, col_list, isotherm_df, pressure_col, loading_col
    ):
        tr_data = self.pressure_bin_features(tr_data, pressure_bins, isotherm_df, pressure_col, loading_col)
        tr_data = self.combine_features(tr_data, feature_list)
        tr_data = self.normalize_df(tr_data, col_list)
        return tr_data

def betml(csv_file, columns=["Pressure","Loading"], verbose=False):

    import importlib
    ml_model = ML()

    isotherm_df = pd.read_csv(csv_file)

    test_data = pd.DataFrame({"name": ["input"]})

    n_bins = 7
    pressure_bins = ml_model.build_pressure_bins(5, 1e5, n_bins)
    feature_list = ["c_%d" % i for i in range(len(pressure_bins))]
    col_list = feature_list + ml_model.generate_combine_feature_list(feature_list)

    test_data = ml_model.add_features(test_data, pressure_bins, feature_list, col_list, isotherm_df, columns[0], columns[1])
  
    empty_bins = [i for i in range(n_bins) if np.isnan(test_data.iloc[0].iloc[i + 1])]
    empty_bin_boundaries = [pressure_bins[i] for i in empty_bins]

    if test_data.isnull().values.any():
        print( f"Missing data in pressure bins: {empty_bin_boundaries}. Prediction failed.")

    test_data = test_data.dropna(subset=col_list)


    with importlib.resources.path('SESAMI', 'lasso_model.sav') as model_path:
        lasso = pickle.load(open(model_path, "rb"))

    test_data["FittedValues"] = lasso.predict(test_data[col_list].values)

    if verbose:
        print(f"Predicted surface area: {test_data.iloc[0]['FittedValues']:.2f} m^2/g")

    return test_data.iloc[0]['FittedValues']