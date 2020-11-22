import math
import random
from horology import Timing
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import seaborn as sns
from zignor import zignor as zigg


def uniform_distribution(a, b):
    random_values = np.random.uniform(a, b, 10000)
    uniform_res = stats.uniform.cdf(random_values, a, b)
    plt.plot(random_values, uniform_res)
    plt.title("Равномерное распределение")
    plt.show()
    print('Равномерно распределенная величина: ', random.uniform(a, b))


def uniform_density(a, b):
    arr = np.random.uniform(a, b, 10000)
    plt.hist(arr)
    plt.title("График плотности равномерного распределения")
    plt.show()


def normal_distribution():
    x = np.linspace(-10, 10, 100)
    y = stats.norm.cdf(x)

    plt.plot(x, y)

    plt.title('Функция нормального распределения')

    plt.show()
    print('Нормально распределенная величина: ', random.uniform(0, 1))


def normal_density1():
    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.title("График плотности нормального распределения")
    plt.show()


def normal_density2(plt_title="График плотности нормального распределения"):
    data = stats.norm.rvs(10.0, 2.5, size=500)
    mu, std = stats.norm.fit(data)
    plt.hist(data, bins=25, density=True, alpha=0.6, color='g')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title(plt_title)
    plt.show()


def exponential_distribution():
    mean = 5
    loc = 0
    xvalues = np.linspace(stats.expon.ppf(0.01, loc, mean), stats.expon.ppf(0.99, loc, mean), 100)
    cdf = stats.expon.cdf(xvalues, loc, mean)
    plt.plot(xvalues, cdf)
    plt.title("Экспоненциальное распределение")
    plt.show()
    print('Экспоненциально распределенная величина: ', random.uniform(0, 1))


def exponential_density():
    sns.distplot(np.random.exponential(size=1000), hist=True)
    plt.title("График плотности экспоненциального распределения")
    plt.show()


def ziggurat():
    x = zigg(50000)
    return x


def muller():
    x = np.linspace(-5, 5, 50000)
    return [x, (1/(np.sqrt(2*np.pi)))*(np.power(np.e, -(np.power(x, 2)/2)))]


def main():
    uniform_distribution(1, 10)
    uniform_density(1, 10)

    normal_distribution()
    normal_density1()
    normal_density2()

    exponential_distribution()
    exponential_density()

    with Timing(name='Ziggurat Algorithm: '):
        values = ziggurat()
    sns.distplot(values)
    plt.title("Ziggurat")
    plt.show()
    with Timing(name='Muller Algorithm: '):
        x, y = muller()
    sns.distplot(ziggurat(), color='red')
    plt.title("Muller")
    plt.show()


if __name__ == "__main__":
    main()


