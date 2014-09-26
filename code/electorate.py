"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import numpy
import thinkbayes2
import thinkplot


class Electorate(thinkbayes2.Suite):
    """Represents hypotheses about the state of the electorate."""

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: the percentage of voters who you hypothesize to favor your candidate
        data: tuple (mean, standard deviation, measurement)
        """
        mean, std_dev, measurement = data
        error = measurement - hypo
        likelihood = thinkbayes2.EvalNormalPdf(error, mean, std_dev)
        return likelihood


def main():
    hypos = numpy.linspace(0, 100, 101)
    suite = Electorate(hypos)

    thinkplot.Pdf(suite, label='prior')

    data = 1.1, 3.7, 53
    suite.Update(data)

    thinkplot.Pdf(suite, label='posterior')
    thinkplot.Show()

    print(suite.Std())
    print(suit.Mean())
    print(suite.ProbLess(50))


if __name__ == '__main__':
    main()
