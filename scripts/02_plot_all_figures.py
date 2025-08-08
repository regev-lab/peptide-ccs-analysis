import matplotlib.pyplot as plt

from peptide_ccs_analysis.figures import plotting as figs

for plot_figure_x in [
    figs.plot_figure_1a_left,
    figs.plot_figure_1a_right,
    figs.plot_figure_1c,
    figs.plot_figure_1d,
    figs.plot_figure_1f,
    figs.plot_figure_2,
    figs.plot_figure_3_and_4abc,
    figs.plot_figure_4de,
    figs.plot_supplementary_figure_s1,
    figs.plot_supplementary_figure_s2,
    figs.plot_supplementary_figure_s3,
    figs.plot_supplementary_figure_s4,
    figs.plot_supplementary_figure_s5,
]:
    plot_figure_x()
    plt.show(block=True)

print('Finished plotting all figures to matplotlib\'s default back-end.')