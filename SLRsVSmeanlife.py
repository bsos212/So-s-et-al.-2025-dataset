from scipy.spatial import ConvexHull
#from itertools import pairwise
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import yaml
matplotlib.use('agg')

def pairwise(iterable):
    iterator = iter(iterable)
    a = next(iterator, None)
    for b in iterator:
        yield a, b
        a=b


def readIsotopes(filename):
    """
    Retrieve the isotopic information
    """

    isotopes = {}
    with open(filename, "r") as fread:
        for document in yaml.load_all(fread, Loader=yaml.Loader):
            isotopes[document["name"]] = document

    return isotopes


class PlotData():

    def __init__(self, axarr):

        self.axarr = axarr

        min_tau = 0.98e5
        max_tau = 1.01e9
        max_points = int(1e5)

        # Min and max tau
        self.tau = np.arange(min_tau, max_tau, max_points)

        # The age of the Milky way in years
        self.t_gal = 8.5e9

        # Values of K [K_min, K_best, K_max] for stable, Pu and Cm
        self.ks = [1.6, 2.3, 5.7, 2.3]

        # Isolation time in years for the free decay scenario
        # The dt-values will be round numbers (in Ma) in the figure's label
        self.decay_times = np.array([10e6, 20e6, 30e6])

        # Mixing time in years for the mixing scenario
        # The t_mix-values will be round numbers (in Ma) in the figure's label
        self.t_mix = np.array([10e6, 40e6, 100e6])

        # The colours of the decay/mixing lines
        self.colors = ['darkblue', 'mediumblue', 'cornflowerblue','cornflowerblue']

        # Tau values to plot
        # 1000 equally space points in logspace, starting at 1e6 and with a
        # multiplicative step of ~3.16 each time (10 ** 0.5), 5 times
        # This reproduces the values from Cote et al. 2021
        n_points = 1000

        tau_decay_log = [6, 6.5, 7, 7.5, 8, 8.5]
        self.tau_decay = [np.logspace(a, b, n_points) for a, b in
                          pairwise(tau_decay_log)]

        # T_iso uncertainties
        # Values from Cote et al. 2021
        t_unc_up = [1.52, 1.415, 1.1903, 1.0759, 1.0266, 1.00873]
        self.t_unc_up = [np.linspace(a, b, n_points) for a, b in
                         pairwise(t_unc_up)]
        t_unc_down = [0.47, 0.641, 0.8230, 0.9260, 0.9736, 0.9913]
        self.t_unc_down = [np.linspace(a, b, n_points) for a, b in
                           pairwise(t_unc_down)]

        self.fontsiz = 17

    def steady_state_k(self,j):
        """
        This function plots the steady state solution and the Monte Carlo
        uncertainty region

        Cote et al. 2021 calulated error bars for the steady-state
        with a Monte Carlo approach.
        """

        # Plots the steady state solution for the three values of K
        self.axarr.plot(self.tau, self.tau / self.t_gal * self.ks[j],
                               color='red')

        # The Monte-Carlo filled region

        # Define tau values and associated uncertainties (Table. 4)
        # for a gamma = 1Ma scenario
        tau2 = np.array([316e6, 100e6, 316e5, 100e5, 316e4, 100e4])
        tau_uncern_up = np.array(
            [1.00873, 1.0266, 1.0759, 1.1903, 1.4150, 1.52])
        tau_uncern_down = np.array(
            [0.99130, 0.9736, 0.9260, 0.8230, 0.6471, 0.47])

        # Define the upper and lower taus
        tau_up = tau2 * tau_uncern_up
        tau_down = tau2 * tau_uncern_down

        points = []
        for ks in self.ks:
            k_coef = ks / self.t_gal

            # Set up the upper and lower arrays
            vu = np.array(list(zip(tau2, tau_up * k_coef)))
            vd = np.array(list(zip(tau2, tau_down * k_coef)))

            # Create the space for them
            points.append(np.empty((vu.shape[0] + vd.shape[0], vu.shape[1]),
                                   dtype=vu.dtype))

            # Interleave them
            points[-1][0::2] = vu
            points[-1][1::2] = vd

        hull = [ConvexHull(p) for p in points]
        #for k, ks in enumerate(self.ks):
        self.axarr.fill(points[j][hull[j].vertices, 0], points[j]
                               [hull[j].vertices, 1], 'red', alpha=0.5)

    def plot_scenario(self, scenario, j, isotope=None):
        """
        Plot the scenario (either free decay lines or mixing)

        If isotope is supplied, then a single tau point is plotted
        """
        assert scenario in ["freedec", "mixing"]

        if scenario == "freedec":

            def scenario_evol(decay_time, tau):
                return np.exp(-decay_time / tau)

            use_variable = self.decay_times
            label_head = "$t_{iso}$"

        elif scenario == "mixing":

            def scenario_evol(mix_time, tau):
                rel_mix = mix_time / tau
                return 1 / (1 + 1.5 * rel_mix + 0.4 * rel_mix ** 2)

            use_variable = self.t_mix
            label_head = "$t_{mix}$"

        if isotope is not None:
            one_tau = isotope["mean_life"]
            tau_ref = isotope["reference_mean_life"]
            error = np.array(isotope["scenario_error"])
            use_k = isotope["k"]
            #The fourth subplot needs the same value
            # of K (K_best) as the second subplot
            use_k.append(use_k[1])

            eq_val = self.ks[j] * one_tau / tau_ref
            self.axarr.errorbar([one_tau, one_tau], [eq_val, eq_val],
                                       yerr=eq_val * error,
                                       markersize=6, fmt='rv')

            lines = eq_val * scenario_evol(use_variable, one_tau)
            for j, line in enumerate(lines):
                self.axarr.errorbar([one_tau, one_tau], [line, line],
                                           yerr=line * error,
                                           markersize=6, fmt='v',
                                           color=self.colors[j])

        else:
            # Plot the lines and Monte-Carlo areas for the scenario
            # If doing decay lines, use the values from Cote et al. 2021
            coef = self.ks[j] / self.t_gal
            for j, value in enumerate(use_variable):

                label = f"{label_head} = {int(value * 1e-6)} Ma"
                lines = self.tau * coef * scenario_evol(value, self.tau)
                self.axarr.plot(self.tau, lines, color=self.colors[j],
                                       label=label)

                for tau, vu, vd in zip(self.tau_decay, self.t_unc_up,
                                           self.t_unc_down):

                    line = tau * coef * scenario_evol(value, tau)
                    self.axarr.fill_between(tau, line * vd, line * vu,
                                                   color="lightsteelblue")

    def plot_isotopes_VR(self, isotopes, grouped_isotopes):
        """
        Plot the isotopes contained in the VR group
        """
        color = "black"
        # For each group in iso_type, plot the isotopes
        for isotope_name in grouped_isotopes:

            isotope = isotopes[isotope_name]
            mean_life = isotope["mean_life"]
            abundance = isotope["abundance"]["VR"]
            x_label_mult = isotope["x_label_mult"]["VR"]
            y_label_mult = isotope["y_label_mult"]["VR"]
            label = isotope["label"]["VR"]

            for i in range(len(self.ks)):
                    self.axarr.plot([mean_life, mean_life], abundance,
                                   color=color)
                    self.axarr.text(x_label_mult, y_label_mult, label,
                                   color=color)

    def plot_isotopes(self, group, isotopes, j, grouped_isotopes):
        """
        Plot the isotopes contained in the groups listed in "iso_type"
        """

        fmt = "o"
        color = "black"
        facecolors= color
        if "Trueman_R" in group:
            color = "brown"
            facecolors = color
        if "noMn" in group:
            facecolors = "w"

        markersize = 6

        def get_error(i, n_y_error, y_error, x_error, references, sigma):
            if n_y_error == 1:
                yerr1 = y_error[0] / references[i]
                yerr2 = y_error[0] / references[i]
                xerr1 = x_error[0] * 2 / sigma
                xerr2 = x_error[0] * 2 / sigma
            else:
                yerr1 = y_error[0]
                yerr2 = y_error[1]
                xerr1 = x_error[0] * 2 / sigma
                xerr2 = x_error[1] * 2 / sigma

            return xerr1, xerr2, yerr1, yerr2

        # For each group in iso_type, plot the isotopes
        for isotope_name in grouped_isotopes:

            isotope = isotopes[isotope_name]
            mean_life = isotope["mean_life"]

            # Grab lengths of lists
            n_abundance = len(isotope["abundance"][group])
            n_reference = len(isotope["reference"][group])
            n_y_error = len(isotope["y_err"][group])
            n_x_error = len(isotope["x_err"][group])

            if n_abundance > 1:
                raise NotImplementedError

            assert n_y_error == n_x_error

            # Values
            abundance = isotope["abundance"][group][0]
            references = isotope["reference"][group]
            # Some isotopes have three different reference (ESS) values,
            # for those the fourth number is equal to the second
            # since both subplots are K=best scenario
            if len(references) == 3:
                references.append(references[1])
            y_error = isotope["y_err"][group]
            x_error = isotope["x_err"][group]
            sigma = isotope["x_sigma"]

            add_label = True
            try:
                x_label_mult = isotope["x_label_mult"][group]
                y_label_mult = isotope["y_label_mult"][group]

                label = isotope["label"][group]
            except KeyError:
                add_label = False

            # Do each plot
            for i in range(len(self.ks)):
             if i == j:
                index = i
                if n_reference == 1:
                    index = 0
                value = abundance / references[index]
                xerr1, xerr2, yerr1, yerr2 = get_error(index, n_y_error,
                                                       y_error, x_error,
                                                       references, sigma)

                self.axarr.errorbar(mean_life, value,
                                       yerr=[[yerr1], [yerr2]],
                                       xerr=[[xerr1], [xerr2]],
                                       fmt=fmt, markersize=markersize,
                                       color=color, mfc=facecolors)

                if not add_label:
                    continue

                y_label_pos = value * y_label_mult
                x_label_pos = mean_life * x_label_mult

                self.axarr.text(x_label_pos, y_label_pos, label,
                                   color=color, fontsize = 22)


# plottig
matplotlib.rcParams.update({'font.size': 22})

# Create a plot with mixing or with the decay lines
# (Options: mixing/freedec) (see Soós et al. 2025 Section: Methodology)
scenario = ('mixing')
assert scenario in ["freedec", "mixing"]

# Groups of isotopes to plot, values are from Soós et al. 2025 Table 1
# 's' stands for the s-process isotopes (107Pd, 182Hf and 205Pb).
# 'r' stands for the r-process isotopes (129I, 244Pu and 247Cm),
#   they are marked by grey points.
# 'p' stands for the p-process isotopes (92Nb and 146Sm).
# 'p_Tc' stands for the two Tc isotopes (97Tc and 98Tc),
#   which only have upper limits for their ESS values.
# 's_Cs' stands for the 135Cs isotope, which only has an
#   upper limit for its ESS value.
# 'SN_Trueman_R' and 'SN_Trueman_N13' stand for the production ratios of
#   Trueman et al. 2025 (53Mn nad 60Fe).
# 'SN_Trueman_R_noMn' and 'SN_Trueman_N13_noMn' stand for the production
#   ratios of the same models as of Trueman et al. 2025, but with the yields
#   of SNIa set to zero.
# 'VR' stands for very short-lived SLRs (27Al, 36Cl, 41Ca)
# 'Pu' stands for 244Pu.
# 'Cm' stands for 247Cm.
iso_type = ['SN_Trueman_N13_noMn',
            'SN_Trueman_N13',
            'SN_Trueman_R',
            'SN_Trueman_R_noMn',
            'p', 's','Pu', 'Cm', 'r','VR']

isotopes = readIsotopes("isotope_data.yaml")


# Define some isotopes

# Instantiate the figure
nrows, ncols=2,2

figure = plt.figure(figsize=(19, 25))
figure.subplots_adjust(hspace=0.07)
figure.subplots_adjust(wspace=0.16)
for i in range(1,5):
    axarr = figure.add_subplot(nrows, ncols, i)

    # Instantiate PlotData object
    plotting = PlotData(axarr)

    # Plot steady state solution and Monte Carlo uncertainty region
    plotting.steady_state_k(i-1)

    # Plot scenarios for Pu244 and Cm247, which have to be treated
    # differently because their reference isotopes are not stable. This means that
    # their steady-state equation and their K values are different
    # The results of these calculations are marked by triangles in the figures
    plotting.plot_scenario(scenario,i-1)
    if "Pu" in iso_type:
        plotting.plot_scenario(scenario,i-1, isotopes["Pu244"])
    if "Cm" in iso_type:
        plotting.plot_scenario(scenario,i-1, isotopes["Cm247"])

    # -----------------------------------------------------------------------------
    # From now on the script prints the isotope data
    # The first value is always the x-axis, therefore the mean-life of the
    # given SLR
    # The second value is the ESS abundance ratio over their production values
    # (see Table 1 of Soós et al. 2025)

    # The labels of the isotopes are handled in a different section, as this
    # figure is quite busy and the labels need to be adjusted for each scenario
    # i.e. mixing or free decay...

    # Divide the isotopes in groups
    grouped_isotopes = {}
    for isotope_name in isotopes:
        groups = isotopes[isotope_name]["groups"]
        for group in groups:
            if group in grouped_isotopes:
                grouped_isotopes[group].append(isotope_name)
            else:
                grouped_isotopes[group] = [isotope_name]

    for group in iso_type:
        # We do not plot the Cm here
        if group == "Cm":
            continue
        if group == "VR":
            # Only plots the 'VR' isotopes on the fourth, unzoomed subplot
            if i == 4:
                plotting.plot_isotopes_VR(isotopes, grouped_isotopes[group])
        else:
            plotting.plot_isotopes(group, isotopes, i-1, grouped_isotopes[group])

    # Legend
    if i == 1:
        axarr.legend(ncol=1,fontsize=22)#bbox_to_anchor=(0.5,0.99))

    # The properties of the figure
    axarr.yaxis.set_ticks_position("both")
    axarr.xaxis.set_ticks_position("both")
    axarr.xaxis.set_major_locator(plt.MaxNLocator(8))
    axarr.yaxis.set_major_locator(plt.MaxNLocator(4))
    axarr.xaxis.set_minor_locator(plt.MaxNLocator(60))
    axarr.yaxis.set_minor_locator(plt.MaxNLocator(40))
    axarr.set_yscale('log')
    axarr.set_xscale('log')
    if i == 1:
        axarr.set_title('K$_{min}$=1.6', y=0.94, x=0.15)
        axarr.set_ylabel('(Z$_R$/Z$_{ref}$)/(P$_R$/P$_{ref}$)'
                         , fontsize=22)
        axarr.set_ylim(2e-5, 8e-2)
        axarr.set_xlim(1e6, 2e8)
    elif i == 2:
        axarr.set_title('K$_{best}$=2.3', y=0.94, x=0.15)
        axarr.set_ylim(2e-5, 8e-2)
        axarr.set_xlim(1e6, 2e8)
    elif i == 3:
        axarr.set_title('K$_{max}$=5.7', y=0.94, x=0.15)
        axarr.set_ylabel('(Z$_R$/Z$_{ref}$)/(P$_R$/P$_{ref}$)'
                         , fontsize=22)
        axarr.set_xlabel('Mean life (Years)', fontsize=22)
        axarr.set_ylim(2e-5, 8e-2)
        axarr.set_xlim(1e6, 2e8)
    elif i == 4:
        axarr.set_title('K$_{best}$=2.3', y=0.94, x=0.15)
        axarr.set_xlabel('Mean life (Years)', fontsize=22)
        axarr.set_ylim(1.5e-6, 1.1e-1)
        axarr.set_xlim(1e5, 3e8)


# ------------------------------------------------------------------------------------------------------------------------------------------
# saving of the plot
if scenario == 'mixing':
    plt.savefig('Tiso_mixing_Zoom.pdf', bbox_inches=matplotlib.transforms.Bbox(
        np.array(((1, 2), (17.5, 22.5)))))
        #np.array(((-0.5, 2), (6.7, 22.5)))))
    print('Plot with the mixing scenario (Clayton 1983) printed')
elif scenario == 'freedec':
    plt.savefig('Tiso_freedec_Zoom.pdf',
                bbox_inches=matplotlib.transforms.Bbox(
                    np.array(((1, 2), (17.4, 22.5)))))
                    #np.array(((-0.5, 2), (6.7, 22.5)))))
    print('Plot with the free decay scenario printed')
