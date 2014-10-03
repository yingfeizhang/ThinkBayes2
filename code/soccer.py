"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import numpy
import thinkbayes2
import thinkplot


class Soccer(thinkbayes2.Suite):
    """Represents hypotheses about."""

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: number of goals per game
        data: a time between goals in minutes
        """
        x = data
        lam = hypo / 90 # convert to goals per minute
        like = thinkbayes2.EvalExponentialPdf(x, lam)
        return like

    def PredRemaining(self, rem_time, score):
        """Plots the predictive distribution for final number of goals.

        rem_time: remaining time in the game in minutes
        score: number of goals already scored
        """
        # TODO: fill this in
        # lam = goals / game
        lam_total = 0
        for lam, prob in self.Items():
            goals_in_remaining_time = lam * rem_time / 90 # convert to goals in remaining time
            lam_total += lt * prob
        


        pmf = thinkbayes2.MakePoissonPmf(goals_in_remaining_time, 12)
        pmf += score
        thinkplot.Pmf(pmf)
        thinkplot.Show()

def main():
    hypos = numpy.linspace(0, 12, 201)
    suite = Soccer(hypos)

    thinkplot.Pdf(suite, label='prior')
    print('prior mean', suite.Mean())

    suite.Update(11)
    thinkplot.Pdf(suite, label='posterior 1')
    print('after one goal', suite.Mean())

    suite.PredRemaining(90-11, )

    thinkplot.Show()


if __name__ == '__main__':
    main()
