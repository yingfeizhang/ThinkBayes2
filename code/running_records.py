import thinkbayes2
import pandas as pd
import numpy as np
import thinkplot
import matplotlib.pyplot as plt
import pickle
import os.path

csv_path = 'Marathon_world_record_times.csv'
pickle_path = 'pickled_world_record_pmf'


def time_converter(string):
    times = string.split(":")
    hours = int(times[0])
    minutes = int(times[1])
    seconds = int(times[2])
    total_hours = hours + minutes / 60.0 + seconds / 60.0 / 60.0
    return 26.2 / total_hours  # mph


def generate_records(dates, alpha, beta, sigma):
    """
    Given a range of dates, an x-intercept, and a slope, generate some running records
    """
    return [alpha + beta * date.value + np.random.normal(scale=sigma) for date in dates]


class RunningRecords(thinkbayes2.Suite, thinkbayes2.Joint):
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
            total_likelihood *= (thinkbayes2.EvalNormalPdf(record_mph, mu=mu, sigma=sigma) + 5)
        return total_likelihood

    def Update(self, data):
        """
        In addition to updating the hypos with the given data, save the data so we can pickle it with this object.
        """
        self.dataframe = data
        return super(RunningRecords, self).Update(data)


def dataframe_to_lists(dataframe):
    dates = [row[1].value for i, row in enumerate(dataframe.values)]
    records = [row[0] for i, row in enumerate(dataframe.values)]
    return dates, records


def create_pmf_from_csv(csv_path):
    """
    Given a path to the running records csv, create a PMF, update it with data, and return it..

    The PMF is a representation of hypotheses for the lines of best fit for the data.
    """
    running_records_dataframe = pd.read_csv(csv_path,
                                            usecols=["Date", "Time"],
                                            parse_dates=["Date"],
                                            converters={"Time": time_converter},
                                            nrows=48,
                                            infer_datetime_format=True,
                                            skiprows=[1])
    male_marathon_records_dataframe = running_records_dataframe[31:]

    dates, records = dataframe_to_lists(male_marathon_records_dataframe)

    slope_estimate = (records[-1] - records[0]) / (dates[-1] - dates[0])

    alphas = np.linspace(11.9, 12.3, 20)
    slopes = np.linspace(slope_estimate / 2.0, slope_estimate * 1.35, 20)
    sigmas = np.linspace(0.01, 0.15, 20)

    hypos = [(alpha, beta, sigma) for alpha in alphas for beta in slopes for sigma in sigmas]
    records_pmf = RunningRecords(hypos)
    records_pmf.Update(male_marathon_records_dataframe)

    with open(pickle_path, 'w') as f:
        pickle.dump(records_pmf, f)
    return records_pmf


def load_pmf_from_pickle(pickle_path):
    """
    Given a path to a pickle file, load the records pmf from it and return the object.
    """
    with open(pickle_path, 'r') as f:
        return pickle.load(f)


if __name__ == "__main__":
    if os.path.isfile(pickle_path):
        records_pmf = load_pmf_from_pickle(pickle_path)
    else:
        records_pmf = create_pmf_from_csv(csv_path)

    dates, records = dataframe_to_lists(records_pmf.dataframe)

    maximum_likelihoods = [0, 0, 0]
    for i in range(3):
        maximum_likelihoods[i] = records_pmf.Marginal(i).MaximumLikelihood()
        print(maximum_likelihoods[i])
        # thinkplot.Hist(records_pmf.Marginal(i))
        # plt.show()


    date_range = pd.date_range(start='12/1/1970', end='1/1/2060', freq='365D')

    for i in range(100):
        alpha, beta, sigma = records_pmf.Random()
        simulated_records = generate_records(date_range, alpha, beta, sigma)
        plt.plot([date.value for date in date_range], simulated_records, 'or')

    plt.plot(dates, records, 'o')
    plt.plot([0, dates[-1]], [maximum_likelihoods[0], maximum_likelihoods[0] + maximum_likelihoods[1] * dates[-1]])

    plt.show()