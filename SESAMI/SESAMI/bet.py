import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib as mpl
from matplotlib import ticker
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import importlib.resources as pkg_resources


mpl.use("TkAgg")
mpl.rcParams["mathtext.default"] = "regular" 
pd.set_option("display.max_rows", 500)


class BETAn:
    def __init__(self, selected_gas, selected_temperature, minlinelength, plotting_information):
        self.R = 8.314
        self.N_A = 6.023e23
        self.T = selected_temperature
        if selected_gas == "Argon":
            self.selected_gas_cs = 0.142e-18
        elif selected_gas == "Nitrogen":
            self.selected_gas_cs = 0.162e-18
        elif selected_gas == "Custom":
            self.selected_gas_cs = float(plotting_information['custom cross section']) * 1e-20 
        self.loadunits = "mol/kg"
        self.minlinelength = (minlinelength)
        self.R2cutoff = plotting_information["R2 cutoff"]
        self.R2min = plotting_information["R2 min"]
        self.eswminima = None
        self.con1limit = None

    def prepdata(self, data, pressure_col="Pressure", loading_col="Loading", conv_to_molperkg=1, p0=1e5, full=True):
        data = data.copy(deep=True)
        data["P_rel"] = data[pressure_col] / p0
        data.sort_values("P_rel", inplace=True)
        data["Loading"] = data[loading_col] * conv_to_molperkg
        data["BETy"] = data["P_rel"] / (data["Loading"] * (1 - data["P_rel"]))
        data["BET_y2"] = data["Loading"] * (1 - data["P_rel"])
        data["phi"] = (data["Loading"] / 1000 * self.R * self.T * np.log(data["P_rel"]))
        if (full):
            self.con1limit = self.getlocalextremum(data, column="BET_y2", x="P_rel", how="Maxima", which=0, points=3)[0]
            self.eswminima = self.getlocalextremum(data, column="phi", x="P_rel", how="Minima", which=0, points=3)[0]
        return data

    def getlocalextremum(self, data, column=None, x=None, how="Minima", which=0, points=3):
        data = data.copy(deep=True)
        start = data.index.values[0]
        end = data.index.values[-1]
        if column is None:
            target = data.columns[0]
        if type(column) == int:
            target = data.columns[column]
        if type(column) == str:
            target = column
        data["target"] = data[target]
        if how == "Maxima":
            data["target"] = -data["target"]
        if x is None:
            data["x"] = data.index
        elif type(x) == int:
            data["x"] = data[data.columns[x]]
        elif type(x) == str:
            data["x"] = data[x]
        else:
            raise ValueError
        data["slopes"] = 0.0
        points = int(points) 
        for i in np.arange(start + points, end - points + 1, 1):
            regdata = data[i - points : i + points + 1][["target", "x"]]
            res = smf.ols("target ~ x", regdata).fit()
            slope = res.params.iloc[1]
            data.at[i, "slopes"] = slope
        minimas = data.index[(data["slopes"].shift(1).fillna(0) < 0) & (data["slopes"].shift(-1).fillna(0) > 0)].values
        goodminimas = []
        if minimas.shape[0] != 0:
            for minimap in minimas:
                if (data[minimap - points : minimap]["target"].mean()
                    > data[data.index == minimap]["target"].values[0]
                    and data[minimap + 1 : minimap + points + 1]["target"].mean()
                    > data[data.index == minimap]["target"].values[0]):
                    goodminimas.append(minimap)
            if goodminimas != []:
                if type(which) == int:
                    minima = goodminimas[which]
                    targetvalue = data[data.index == minima]["target"].values[0]
                if type(which) == list:
                    minima = [goodminimas[minima] for minima in which]
                    targetvalue = [data[data.index == minima]["target"].values[0] for minima in minima ]
            else:
                minima = None
                targetvalue = None
        else:
            minima = None
            targetvalue = None
        return minima, targetvalue, data[["x", "target", "slopes"]]

    def th_loading(self, x, params):
        [qm, C] = params
        bet_y = (C - 1) / (qm * C) * x + 1 / (qm * C)
        bet_loading = x / (bet_y * (1 - x))
        return bet_loading

    def gen_phi(self, load, p_rel, T=87.0):
        phi = load / 1000 * 8.314 * T * np.log(p_rel)
        return phi

    def makeisothermplot(self, plotting_information, ax, data, yerr=None, tryminorticks="Yes", xscale="log", fit_data=None):
        ax.errorbar(data["P_rel"], data["Loading"], yerr=yerr, fmt="o", capsize=3, label="Isotherm data points")
        ax.xaxis.label.set_text("$p/p_0$")
        ax.yaxis.label.set_text("$q$" + " / " + "$%s$" % self.loadunits)
        if xscale == "log":
            ax.set_xscale("log")
            if tryminorticks == "Yes":
                locmaj = mpl.ticker.LogLocator(base=10.0, numticks=10)
                ax.xaxis.set_major_locator(locmaj)
                locmin = mpl.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=10)
                ax.xaxis.set_minor_locator(locmin)
                ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        if xscale == "linear":
            ax.set_xlim((0, 1))
            ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_ylim((0, ax.get_ylim()[1]))
        [bet_info, betesw_info] = fit_data
        [rbet, bet_params] = bet_info
        [rbetesw, betesw_params] = betesw_info
        if rbet != (None, None):
            ax.axvspan(data.at[rbet[0], "P_rel"],
                data.at[rbet[1], "P_rel"],
                facecolor=plt.cm.PuOr(70),
                edgecolor="none",
                alpha=0.6,
                label="BET region")
            ax.plot(data["P_rel"].values,
                self.th_loading(data["P_rel"].values, bet_params),
                color=plt.cm.PuOr(20),
                label="BET fit")
        if rbetesw != (None, None):
            ax.axvspan(data.at[rbetesw[0], "P_rel"],
                data.at[rbetesw[1], "P_rel"],
                facecolor=plt.cm.Greens(70),
                edgecolor="none",
                alpha=0.6,
                label="BET+ESW region")
            ax.plot(data["P_rel"].values,
                self.th_loading(data["P_rel"].values, betesw_params),
                color=plt.cm.Greens(200),
                label="BET+ESW fit")
        bet_values = [data["Loading"].values[i]
            for i, value in enumerate(data["P_rel"].values)
            if ax.get_xlim()[0] <= value <= ax.get_xlim()[1]]
        y_max = max(bet_values) + 10
        ax.set_ylim(top=y_max)
        if self.eswminima is not None:
            ax.vlines(data.at[self.eswminima, "P_rel"],
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                colors=plt.cm.Greens(200),
                linestyles="dashed",
                label="First ESW minimum")
        if self.con1limit is not None:
            ax.vlines(data.at[self.con1limit, "P_rel"],
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                linestyles="dashed",
                color=plt.cm.Purples(230),
                label="Consistency 1 maximum")
        titletext = "Isotherm Data"
        ax.set_title(titletext)
        if plotting_information["legend"] == "Yes":
            ax.legend(loc="upper left")

    def makeconsistencyplot(self, plotting_information, ax3, data, tryminorticks):
        ax3.xaxis.label.set_text("$p/p_0$")
        ax3.yaxis.label.set_text("$q(1-p/p_{0})$" + " / " + "$%s$" % self.loadunits)
        ax3.set_xscale("log")
        titletext = "BET Consistency Plot"
        ax3.set_title(titletext)
        ax3.errorbar(data["P_rel"], data["BET_y2"], fmt="o", label="Isotherm data points")
        ax3.set_ylim(ax3.get_ylim())
        if tryminorticks == "Yes":
            locmaj = mpl.ticker.LogLocator(base=10.0, numticks=10)
            ax3.xaxis.set_major_locator(locmaj)
            locmin = mpl.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=10)
            ax3.xaxis.set_minor_locator(locmin)
            ax3.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ind_max = self.con1limit
        if ind_max is not None:
            x_max = data[data.index == ind_max]["P_rel"].values[0]
            ax3.vlines(x_max, ax3.get_ylim()[0], ax3.get_ylim()[1], colors=plt.cm.Purples(230), linestyles="dashed", label="Consistency 1 maximum")
        else:
            x_max = data["P_rel"][data["BET_y2"].idxmax()]
        if self.eswminima is not None:
            ax3.vlines(data.at[self.eswminima, "P_rel"], ax3.get_ylim()[0], ax3.get_ylim()[1], colors=plt.cm.Greens(200), linestyles="dashed", label="First ESW minimum")
        ax3.set_xlim(right=1.000)
        if plotting_information["legend"] == "Yes":
            ax3.legend(loc="upper left")

    def makelinregplot(self, plotting_information, ax2, p, q, data, mode, columns):
        bbox_props = dict(boxstyle="square", ec="k", fc="w", lw=1.0)
        if (p, q) == (None, None):
            ax2.text(0.97, 0.22, "No suitable linear region found.", horizontalalignment="right", verticalalignment="center", bbox=bbox_props, transform=ax2.transAxes)
        else:
            [linear, stats, C, qm, _, _, _, _, _, con3, con4, A_BET] = self.linregauto(p, q, data)
            [_, _, _, _, _, _, results] = stats
            intercept, slope = results.params
            low_p = data.iloc[p][columns[0]]
            high_p = data.iloc[q][columns[0]]
            ax2.xaxis.label.set_text("$p/p_0$")
            ax2.yaxis.label.set_text(r"$\frac{p/p_0}{q(1-p/p_0)}$" + " / " + "kg/mol")
            if mode == "BET":
                titletext = "BET Linear Region Plot"
            elif mode == "BET+ESW":
                titletext = "BET+ESW Linear Region Plot"
            ax2.set_title(titletext)
            ax2.errorbar(linear["P_rel"], linear["BETy"], fmt="o", label="BET data points")
            ax2.plot(linear["P_rel"], slope * linear["P_rel"] + intercept, "k", alpha=0.5, label="Fitted Linear Region")
            ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
            ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)
            plt.setp(ax2.yaxis.get_majorticklabels(), rotation=30)
            if plotting_information["legend"] == "Yes":
                ax2.legend()
        my_dict = {"C": C, "qm": qm, "A_BET": A_BET, "con3": con3, "con4": con4, "length_linear_region": q - p, "R2_linear_region": results.rsquared, "low_P_linear_region": low_p, "high_P_linear_region": high_p}
        return my_dict

    def eswdata(self, data, eswpoints):
        data = data.copy(deep=True)
        data["phi"] = (data["Loading"] / 1000 * self.R * self.T * np.log(data["P_rel"])) 
        if self.eswminima is None:
            minima = self.getlocalextremum(data, column="phi", x="P_rel", how="Minima", which=0, points=eswpoints)[0]
        else:
            minima = self.eswminima
        if minima is not None:
            eswarea = (data[data.index == minima]["Loading"].values[0] / 1000 * self.N_A * self.selected_gas_cs)
        else:
            eswarea = None
        return [data["Loading"], data["phi"], minima, eswarea]

    def makeeswplot(self, plotting_information, ax, data, eswpoints, fit_data):
        [loading, phi, minima, _] = self.eswdata(data, eswpoints)
        ax.errorbar(loading, phi, yerr=None, fmt="o", capsize=3, label="Isotherm data points")
        ax.set_ylim(ax.get_ylim())
        ax.xaxis.label.set_text("q / mol/kg")
        ax.yaxis.label.set_text("$\\Phi$ / J/g")
        ax.set_title("ESW Plot")
        ax.set_xlim((0, ax.get_xlim()[1]))
        bbox_props = dict(boxstyle="square", ec="k", fc="w", lw=1.0)
        [bet_info, betesw_info] = fit_data
        [rbet, bet_params] = bet_info
        [rbetesw, betesw_params] = betesw_info
        if rbet != (None, None):
            ax.axvspan(data.at[rbet[0], "Loading"], data.at[rbet[1], "Loading"], facecolor=plt.cm.PuOr(70), edgecolor="none", alpha=0.6, label="BET region")
            load_bet = self.th_loading(data["P_rel"].values, bet_params)
            ax.plot(load_bet, self.gen_phi(load_bet, data["P_rel"].values), color=plt.cm.PuOr(20), label="BET fit")
        if rbetesw != (None, None):
            ax.axvspan(data.at[rbetesw[0], "Loading"], data.at[rbetesw[1], "Loading"], facecolor=plt.cm.Greens(70), edgecolor="none", alpha=0.6, label="BET+ESW region")
            load_betesw = self.th_loading(data["P_rel"].values, betesw_params)
            ax.plot(load_betesw, self.gen_phi(load_betesw, data["P_rel"].values), color=plt.cm.Greens(200), label="BET+ESW fit")
        phi_values = [phi[i] for i, value in enumerate(loading) if ax.get_xlim()[0] <= value <= ax.get_xlim()[1]]
        y_min = min(phi_values) - 10
        y_max = max(phi_values) + 10
        ax.set_ylim(bottom=y_min, top=y_max)
        if minima is not None:
            ax.vlines(data.at[minima, "Loading"], ax.get_ylim()[0], ax.get_ylim()[1], colors=plt.cm.Greens(200), linestyles="dashed", label="First ESW minimum")
        else:
            ax.text(0.03, 0.90, "Minima not found", horizontalalignment="left", verticalalignment="center", transform=ax.transAxes, bbox=bbox_props)
        if self.con1limit is not None:
            ax.vlines(data.at[self.con1limit, "Loading"], ax.get_ylim()[0], ax.get_ylim()[1], colors=plt.cm.Purples(230), linestyles="dashed", label="Consistency 1 maximum")
        if plotting_information["legend"] == "Yes":
            ax.legend(loc="upper center")

    def linregauto(self, p, q, data):
        data = data.copy(deep=True)
        linear = data[p:q]
        results = smf.ols("BETy ~ P_rel", linear).fit()
        intercept, slope = results.params
        ftest = (results.fvalue, results.f_pvalue)
        Ttest = results.t_test(np.array([[1, 0], [0, 1]]))
        ttest = (Ttest.tvalue, Ttest.pvalue)
        influence = results.get_influence()
        resid_stud = influence.get_resid_studentized_external()
        prel = linear["P_rel"].values
        bety = linear["BETy"].values
        isoutlier = "No"
        preloutlier = False
        betyoutlier = False
        if np.absolute(resid_stud).max() > 3.0:
            isoutlier = "Yes"
            arrindexoutlier = np.where(resid_stud > 3.0)[0]
            preloutlier = prel[arrindexoutlier]
            betyoutlier = bety[arrindexoutlier]
        outlierdata = [isoutlier, preloutlier, betyoutlier]
        norm_res = (resid_stud - resid_stud.mean()) / resid_stud.std()
        shaptest = ss.shapiro(norm_res)
        r2 = results.rsquared
        r2adj = results.rsquared_adj
        stats = [ftest, ttest, outlierdata, shaptest, r2, r2adj, results]
        if intercept == 0.0:
            intercept += 1e23
        C = slope / intercept + 1
        qm = 1 / (slope + intercept)
        ind_max = self.con1limit
        if ind_max is not None:
            x_max = data[data.index == ind_max]["P_rel"].values[0]
        else:
            x_max = data["P_rel"][data["BET_y2"].idxmax()]
        if linear["P_rel"].max() <= x_max:
            con1 = "Yes"
        else:
            con1 = "No"
        if C > 0:
            con2 = "Yes"
        else:
            con2 = "No"
        lower_limit_y = data["Loading"][data["Loading"] <= qm].max()
        upper_limit_y = data["Loading"][data["Loading"] > qm].min()
        lower_limit_x = data["P_rel"][data["Loading"] <= qm].max()
        upper_limit_x = data["P_rel"][data["Loading"] > qm].min()
        m = (upper_limit_y - lower_limit_y) / (upper_limit_x - lower_limit_x)
        x_BET3 = upper_limit_x - (upper_limit_y - qm) / m
        if linear["P_rel"].min() <= x_BET3 <= linear["P_rel"].max():
            con3 = "Yes"
        else:
            con3 = "No"
        x_BET4 = 1 / (np.sqrt(C) + 1)
        if np.abs((x_BET4 - x_BET3) / x_BET3) < 0.2:
            con4 = "Yes"
        else:
            con4 = "No"
        A_BET = qm * self.N_A * self.selected_gas_cs / 1000
        return [linear, stats, C, qm, x_max, x_BET3, x_BET4, con1, con2, con3, con4, A_BET]

    def picklen(self, data, method="BET+ESW"):
        data_og = data.copy(deep=True)
        iddatamax = self.con1limit
        if iddatamax is None:
            iddatamax = data["BET_y2"].idxmax()
        data = data_og[: (iddatamax + 2)].copy(deep=True)
        minlength = int(self.minlinelength - 1)
        R2cutoff = self.R2cutoff
        start = data.index.values[0]
        end = data.index.values[-1]
        curbest = [None, None, -1, 1, 0.0]
        satisflag = (0)
        endlowlimit = start + minlength
        starthighlimit = end - minlength
        if method == "BET+ESW":
            minima = self.eswminima
            if minima is not None:
                endlowlimit = minima + 1
                starthighlimit = (minima - 1)
        for i in np.arange(end, endlowlimit, -1):
            for j in np.arange(start, i - minlength, 1):
                p, q = j, i
                if p > starthighlimit:
                    continue
                if q - p > 10:
                    continue
                [_, stats, _, _, _, _, _, con1, con2, con3, con4, _] = self.linregauto(p, q, data)
                [ftest, ttest, _, shaptest, r2, _, _] = stats
                if (con1 == "Yes" and con2 == "Yes" and ftest[1] < 0.99 and ttest[1].max() < 0.99 and shaptest[1] > 0.01 and r2 > self.R2min):
                    scon3 = int(1) if con3 == "Yes" else int(0)
                    scon4 = int(1) if con4 == "Yes" else int(0)
                    conscore = (scon3 + scon4)
                    length = q - p
                    R2 = r2
                    if conscore == int(2) and length > minlength and R2 > R2cutoff:
                        curbest = [p, q, conscore, length, R2]
                        satisflag = 1
                        break
                    if curbest[2] == -1:
                        curbest = [p, q, conscore, length, R2]
                    if conscore > curbest[2]:
                        curbest = [p, q, conscore, length, R2]
            if satisflag == 1:
                break
        fp, fq, _, _, _ = curbest
        return fp, fq

    def saveimgsummary(self, plotting_information, bet_info, betesw_info, data, eswminima, columns):
        rbet = bet_info[0]
        rbetesw = betesw_info[0]
        fig1, fig2, fig3, fig4, fig5 = (plt.figure(), plt.figure(), plt.figure(), plt.figure(), plt.figure())
        ax, ax2, ax3, ax4, ax5 = ( fig1.add_subplot(111), fig2.add_subplot(111), fig3.add_subplot(111), fig4.add_subplot(111), fig5.add_subplot(111))
        self.makeisothermplot(plotting_information, ax, data, None, "Yes", "log", [bet_info, betesw_info])
        self.makelinregplot(plotting_information, ax2, rbet[0], rbet[1], data, "BET", columns)
        self.makeconsistencyplot(plotting_information, ax3, data, "Yes")
        self.makeeswplot(plotting_information, ax4, data, 3, [bet_info, betesw_info])
        self.makelinregplot(plotting_information, ax5, rbetesw[0], rbetesw[1], data, "BET", columns)

        dpi = plotting_information["dpi"]
        if plotting_information["save fig"]:
            fig_names = ["isotherm.png","BETPlotLinear.png","BETPlot.png","ESWPlot.png","BETESWPlot.png"]
            for i,fig in enumerate([fig1, fig2, fig3, fig4, fig5]):
                fig.savefig(fig_names[i], format="png", dpi=dpi, bbox_inches="tight")
                plt.close(fig)
        figf = plt.figure(figsize=(3 * 7.0, 2 * 6.0))
        [[_, _, _], [ax2f, ax5f, _]] = figf.subplots(nrows=2, ncols=3)
        BET_dict = self.makelinregplot(plotting_information, ax2f, rbet[0], rbet[1], data, "BET", columns)
        BET_ESW_dict = None
        if eswminima is None:
            ax5f.axis("off")
        else:
            BET_ESW_dict = self.makelinregplot(plotting_information, ax5f, rbetesw[0], rbetesw[1], data, mode="BET+ESW", columns=columns)
        return BET_dict, BET_ESW_dict

    def generatesummary(self, data, plotting_information, eswpoints, columns):
        with pkg_resources.path('SESAMI', 'mplstyle.txt') as style_path:
            plt.style.use(style_path)
        [_, _, eswminima, _] = self.eswdata(data, eswpoints)
        p, q = self.picklen(data, method="BET") 
        rbet = (p, q)
        if rbet == (None, None):
            return ('BET linear failure', 'BET linear failure') 
        else:
            (p, q) = rbet
            [_, _, C, qm, _, _, _, _, _, _, _, _] = self.linregauto(p, q, data)
            bet_params = (qm, C)
        betesw_info = None
        if eswminima is None:
            rbetesw = (None, None)
            return 'No eswminima', 'No eswminima'
        else:
            p, q = self.picklen(data, method="BET+ESW")
            rbetesw = (p, q)
        if rbetesw == (None, None):
            return ('BET+ESW linear failure', 'BET+ESW linear failure')
        else:
            (p, q) = rbetesw
            [_, _, C, qm, _,  _, _, _, _, _, _, _] = self.linregauto(p, q, data)
            betesw_params = (qm, C)
        betesw_info = [rbetesw, betesw_params]
        bet_info = [rbet, bet_params]
        mpl.rcParams.update({"font.size": plotting_information["font size"]})
        mpl.rcParams["font.family"] = plotting_information["font type"]
        BET_dict, BET_ESW_dict = self.saveimgsummary(plotting_information, bet_info, betesw_info, data, eswminima, columns)

  
        return BET_dict, BET_ESW_dict

def convert_numpy_scalars(d):
    new_d = {}
    for key, value in d.items():
        if isinstance(value, (np.integer, np.floating)):
            new_d[key] = value.item()
        else:
            new_d[key] = value
    return new_d

def fitbet(csv_file, columns=["Pressure","Loading"],
                    adsorbate="N2", p0=1e5, T=77,
                    R2_cutoff=0.9995, R2_min=0.998,
                    font_size=12, font_type="DejaVu Sans",
                    legend=True, dpi=600, save_fig=True, verbose=False):
    
    setting = {}
    minlinelength = 4
    if adsorbate != 'N2' and adsorbate != 'Ar' :
        setting["custom saturation pressure"] = float(p0)
        setting["custom temperature"] = float(T)
        setting['custom adsorbate'] = 'No'
        gas = 'Custom'
        temperature = float(T)
    elif adsorbate == 'N2':
        setting["custom saturation pressure"] = float(1e5)
        setting["custom temperature"] = float(77)
        setting["gas"] = 'Nitrogen'
        temperature = 77
        gas = 'Nitrogen'
    elif adsorbate == 'Ar':
        setting["custom saturation pressure"] = float(1e5)
        setting["gas"] = 'Argon'
        temperature = 87
        gas = 'Argon'

    setting["font size"] = int(font_size)
    setting["R2 cutoff"] = float(R2_cutoff)
    setting["R2 min"] = float(R2_min)
    setting["dpi"] = float(dpi)
    setting["font type"] = font_type
    setting["save fig"] = save_fig

    if legend:
        setting["legend"] = "Yes"
    else:
        setting["legend"] = "No"
   
    b = BETAn(gas, temperature, minlinelength, setting)
    data = pd.read_csv(csv_file, usecols=columns)
    if data[columns[0]].iloc[0] == 0:
        data.loc[0, columns[0]] = data[columns[0]].iloc[1] / 2
    data = b.prepdata(data, pressure_col=columns[0], loading_col=columns[1], p0=p0)
    BET_dict, BET_ESW_dict = b.generatesummary(data, setting, 3, columns)

    if verbose:
        print("*"*75)
        print("BET result")
        print("-"*75)
        print("BET Suface Area:",BET_dict["A_BET"],"mm2/g")
        print("Fitting points:",BET_dict["length_linear_region"])
        print("Fitting parameter","C:",BET_dict["C"],"qm:",BET_dict["qm"])
        print("Fitting region","C:","low region: ",BET_dict["low_P_linear_region"],"Pa", "high regio:",BET_dict["high_P_linear_region"],"Pa")
        print("con3:",BET_dict["con3"],"con4:",BET_dict["con4"])
        print("Fitting R2",BET_dict["R2_linear_region"])
        print("*"*75)
        print("BET + ESW result")
        print("-"*75)
        print("BET Suface Area:",BET_ESW_dict["A_BET"],"mm2/g")
        print("Fitting points:",BET_ESW_dict["length_linear_region"])
        print("Fitting parameter","C:",BET_ESW_dict["C"],"qm:",BET_ESW_dict["qm"])
        print("Fitting region","C:","low region: ",BET_ESW_dict["low_P_linear_region"],"Pa", "high regio:",BET_ESW_dict["high_P_linear_region"],"Pa")
        print("con3:",BET_ESW_dict["con3"],"con4:",BET_ESW_dict["con4"])
        print("Fitting R2",BET_ESW_dict["R2_linear_region"])
        print("-"*75)

    return convert_numpy_scalars(BET_dict), convert_numpy_scalars(BET_ESW_dict)