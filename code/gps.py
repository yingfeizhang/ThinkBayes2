"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import numpy
import thinkbayes2
import thinkplot

from itertools import product


class Gps(thinkbayes2.Suite, thinkbayes2.Joint):
    """Represents hypotheses about your location in the field."""

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: x, y
        data: x, y
        """
        
        measured_x, measured_y = data
        actual_x, actual_y = hypo

        error_x = measured_x - actual_x
        error_y = measured_y - actual_y

        prob_x = thinkbayes2.EvalNormalPdf(error_x, 0, 30)
        prob_y = thinkbayes2.EvalNormalPdf(error_y, 0, 30)

        like = prob_x * prob_y

        return like


def main():
    coords = numpy.linspace(-100, 100, 101)
    joint = Gps(product(coords, coords))

    joint.Update((51, -15))
    # joint.Update((48, 90))

    pairs = [(11.903060613102866, 19.79168669735705),
             (77.10743601503178, 39.87062906535289),
             (80.16596823095534, -12.797927542984425),
             (67.38157493119053, 83.52841028148538),
             (89.43965206875271, 20.52141889230797),
             (58.794021026248245, 30.23054016065644),
             (2.5844401241265302, 51.012041625783766),
             (45.58108994142448, 3.5718287379754585)]

    # joint.UpdateSet(pairs)

    x_marginal = joint.Marginal(0, label='x')
    y_marginal = joint.Marginal(1, label='y')

    thinkplot.Pdf(x_marginal)
    thinkplot.Pdf(y_marginal)
    thinkplot.Show()

    # TODO: plot the marginals and print the posterior means


if __name__ == '__main__':
    main()
