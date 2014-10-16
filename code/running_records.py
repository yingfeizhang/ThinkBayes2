import thinkbayes2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = 'Marathon_world_record_times.csv'


def time_converter(string):
    times = string.split(":")
    hours = int(times[0])
    minutes = int(times[1])
    seconds = int(times[2])
    total_hours = hours + minutes / 60.0 + seconds / 60.0 / 60.0
    return 26.2 / total_hours  # mph


class WorldRecords(thinkbayes2.Suite, thinkbayes2.Joint):
    def Likelihood(self, data, hypo):
        """
        Calculate the likelihood of a hypo given the data
        data: [{"date":'1998-09-20', "mph": 12.142122}, ...] list of records
        hypo: (alpha, beta, sigma)
        """
        alpha, beta, sigma = hypo
        total_likelihood = 1
        for i, row in enumerate(data.values):
            record_date = row[1].value  # ms
            record_mph = row[0]
            mu = alpha + beta * record_date
            total_likelihood *= thinkbayes2.EvalNormalPdf(record_mph, mu=mu, sigma=sigma)
        return total_likelihood


if __name__ == "__main__":
    world_records = pd.read_csv(path,
                                usecols=["Date", "Time"],
                                parse_dates=["Date"],
                                converters={"Time": time_converter},
                                nrows=48,
                                infer_datetime_format=True,
                                skiprows=[1])

    world_records_clipped = world_records[31:]
    data = world_records_clipped

    dates = [row[1].value for i, row in enumerate(data.values)]
    records = [row[0] for i, row in enumerate(data.values)]

    slope_estimate = (records[-1] - records[0]) / (dates[-1] - dates[0])

    alphas = np.linspace(12.1, 12.2, 10)
    slopes = np.linspace(slope_estimate / 4.0, slope_estimate * 4.0, 10)
    sigmas = np.linspace(0.01, 1, 10)

    hypos = [(alpha, beta, sigma) for alpha in alphas for beta in slopes for sigma in sigmas]
    world_records = WorldRecords(hypos)
    world_records.Update(data)
