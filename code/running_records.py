import pickle
import os.path
import thinkbayes2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import thinkplot

csv_path = 'Marathon_world_record_times.csv'
pmf_pickle_path = 'pickled_world_record_pmf'
simulation_pickle_path = 'pickled_simulation'


def string_time_to_mph(string):
    """
    Convert from string record time to a float mph
    """
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
    return [(date, alpha + beta * date.value + np.random.normal(scale=sigma)) for date in dates]


class RunningRecords(thinkbayes2.Suite, thinkbayes2.Joint):
    def Likelihood(self, data, hypo):
        """
        Calculate the likelihood of a hypo given the data
        data: [(Timestamp, 12.142122), ...] pandas dataframe of records in the form of (date, record in mph) tuples
        hypo: (alpha, beta, sigma)
        """
        alpha, beta, sigma = hypo
        total_likelihood = 1
        for i, row in enumerate(data.values):
            date = row[1].value  # ms
            measured_mph = row[0]
            predicted_mph = alpha + beta * date
            error = measured_mph - predicted_mph
            total_likelihood *= thinkbayes2.EvalNormalPdf(error, mu=0, sigma=sigma)
        return total_likelihood

    def Update(self, data):
        """
        In addition to updating the hypos with the given data, save the data so we can pickle it with this object.
        """
        self.dataframe = data
        return super(RunningRecords, self).Update(data)


def dataframe_to_lists(dataframe):
    """
    Convert from a pandas dataframe to two lists, dates = [date1, date2, date3... ] (ns), records = [12.1, 12.3] (mph)
    """
    dates = [row[1].value for i, row in enumerate(dataframe.values)]
    records = [row[0] for i, row in enumerate(dataframe.values)]
    return dates, records


def create_pmf_from_csv(csv_path):
    """
    Given a path to the running records csv, create a PMF, update it with data, and return it.

    The PMF is a representation of hypotheses for the lines of best fit for the data.
    """
    running_records_dataframe = pd.read_csv(csv_path,
                                            usecols=["Date", "Time"],
                                            parse_dates=["Date"],
                                            converters={"Time": string_time_to_mph},
                                            nrows=48,
                                            infer_datetime_format=True,
                                            skiprows=[1])
    male_marathon_records_dataframe = running_records_dataframe[29:]

    dates, records = dataframe_to_lists(male_marathon_records_dataframe)

    intercept_estimate, slope_estimate = thinkbayes2.LeastSquares(dates, records)

    alpha_range = 0.007
    beta_range = 0.2

    alphas = np.linspace(intercept_estimate * (1 - alpha_range), intercept_estimate * (1 + alpha_range), 40)
    betas = np.linspace(slope_estimate * (1 - beta_range), slope_estimate * (1 + beta_range), 40)
    sigmas = np.linspace(0.001, 0.1, 40)

    hypos = [(alpha, beta, sigma) for alpha in alphas for beta in betas for sigma in sigmas]
    records_pmf = RunningRecords(hypos)
    records_pmf.Update(male_marathon_records_dataframe)

    with open(pmf_pickle_path, 'w') as f:
        pickle.dump(records_pmf, f)
    return records_pmf


def load_pmf():
    """
    Load the records PMF from a file and return it.
    """
    with open(pmf_pickle_path, 'r') as f:
        return pickle.load(f)


def load_simulations():
    """
    Load the simulations list from a file and return it.
    """
    with open(simulation_pickle_path, 'r') as f:
        return pickle.load(f)


def find_closest(lst, num):
    """
    Given a list and a number, return the element in that list closest to the number.

    Use this to fit numbers into discrete buckets.
    """
    return min(lst, key=lambda x:abs(x-num))


def compare_to_least_squares(alpha, beta, dates, records):
    """
    Plot the original data, a guess of alpha + beta * date, and the least squares guess on one plot.
    """
    dates_years = [pd.to_datetime(date) for date in dates]
    plt.plot(dates_years, records, 'o')
    intercept, slope = thinkbayes2.LeastSquares(dates, records)
    plt.plot([dates_years[0], dates_years[-1]], [intercept + slope * dates[0], intercept + slope * dates[-1]])
    plt.plot([dates_years[0], dates_years[-1]], [alpha + beta * dates[0], alpha + beta * dates[-1]])
    plt.legend(["data", "mean squared", "maximum likelihood"], loc='upper left')
    plt.title('Marathon record estimates')
    plt.xlabel('Year')
    plt.ylabel('mph')
    plt.show()

if __name__ == "__main__":

    # compute the PMF of line estimates (or load the cached version)
    if os.path.isfile(pmf_pickle_path):
        records_pmf = load_pmf()
    else:
        records_pmf = create_pmf_from_csv(csv_path)

    dates, records = dataframe_to_lists(records_pmf.dataframe)

    # compute the marginal distributions for alpha, beta, and sigma.
    # save their maximum likelihoods
    maximum_likelihoods = [0, 0, 0]
    for title, i in [('alpha', 0), ('beta', 1), ('sigma', 2)]:
        marginal = records_pmf.Marginal(i)
        maximum_likelihoods[i] = marginal.MaximumLikelihood()
        thinkplot.Hist(marginal)
        plt.title("PMF for " + title)
        plt.show()

    # compare the alpha and beta maximum likelihoods to the least squares estimate
    compare_to_least_squares(maximum_likelihoods[0], maximum_likelihoods[1], dates, records)

    # run a monte-carlo simulation of running records (or load the cached version)
    date_range = pd.date_range(start=dates[-1], end='1/1/2060', freq='365D')

    if os.path.isfile(simulation_pickle_path):
        simulated_records = load_simulations()
    else:
        simulated_records = []
        for i in range(1000):
            alpha, beta, sigma = records_pmf.Random()
            simulated_records.append(generate_records(date_range, alpha, beta, sigma))
        with open(simulation_pickle_path, 'w') as f:
            pickle.dump(simulated_records, f)

    # scatterplot of simulated records
    for simulation in simulated_records:
        plt.plot([item[0] for item in simulation], [item[1] for item in simulation], 'or')
    plt.plot([pd.to_datetime(date) for date in dates], records, 'o')
    plt.title("Simulated records")
    plt.xlabel("year")
    plt.ylabel("mph")
    plt.show()

    # pcolor plot of the simulated records
    joint_estimate = thinkbayes2.Joint()
    for simulation in simulated_records:
        for date, record in simulation:
            joint_estimate.Incr((date.value, round(record, 2)))
    thinkplot.Contour(joint_estimate, contour=False, pcolor=True)
    plt.plot([dates[0], pd.to_datetime('2041').value],[13.1094, 13.1094], 'r')
    plt.plot([pd.to_datetime('2041').value, pd.to_datetime('2041').value], [12, 13.1094], 'r')
    plt.plot(dates, records, 'bo')
    plt.title("simulated records joint pmf")
    thinkplot.show()

    year2041 = find_closest(date_range, pd.to_datetime('2041'))
    year2042 = find_closest(date_range, pd.to_datetime('2042'))

    # calculate some summary statistics
    pmf_2hour = joint_estimate.Conditional(0, 1, 13.11)
    cdf_2hour = pmf_2hour.MakeCdf()
    print("probability of 2-hour marathon by 2041: %f" % pmf_2hour.ProbLess(year2042.value))
    print(pd.to_datetime(cdf_2hour.Value(.385445)))
    print("year when we're 90% confident of having ran a 2-hour marathon " + str(pd.to_datetime(cdf_2hour.Value(.9))))

    # plot percentiles of simulated records
    speeds = np.arange(12.8, 13.2, 0.01)
    percentiles = {.05:[], .25:[], .5:[], .75:[], .95:[]}
    percentile_colors = {.05:'r', .25:'m', .5:'y', .75:'b', .95:'g'}
    for percentile in percentiles:
        for speed in speeds:
            cdf = joint_estimate.Conditional(0, 1, round(speed,2)).MakeCdf()
            percentiles[percentile].append(cdf.Value(percentile))
        plt.plot([pd.to_datetime(date) for date in percentiles[percentile]], speeds, '.',  color=percentile_colors[percentile])
        inter, slope = thinkbayes2.LeastSquares(speeds, percentiles[percentile])
        plt.plot([pd.to_datetime(inter + speeds[0] * slope), pd.to_datetime(inter + speeds[-1] * slope)],[speeds[0], speeds[-1]], label=str(percentile), color=percentile_colors[percentile])

    plt.plot([pd.to_datetime(date) for date in dates], records, 'ob')
    plt.plot([pd.to_datetime(dates[0]), pd.to_datetime('2041')],[13.1094, 13.1094], color='#000000')
    plt.plot([pd.to_datetime('2041'), pd.to_datetime('2041')], [12, 13.1094], color='#000000')
    plt.legend(loc="upper left")
    plt.title("Conditional probability of years summary")
    plt.xlabel("year")
    plt.ylabel("mph")
    plt.show()