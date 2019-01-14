from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, \
    RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle


def timed(func):
    """A simple decorator that prints the elapsed time of the function call."""

    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        print('Elapsed time: %.2fs.' % (time() - start))

        return result

    return wrapper


class Experiment:
    """Runs a series of classification tests on the bacteria fluorescence
    dataset.
    """
    growth_phases = ['lag', 'log', 'stat']
    integration_times = ['16ms', '32ms']

    def __init__(self, growth_phases='all', n_jobs=-1, random_seed=42):
        """Create an experiment that gets classification scores with various
        setups.

        Arguments:
            growth_phases: Which growth_phases of the dataframe to use.
            Must be one of 'lag', 'log', 'stat', or 'all'. Defaults to 'all'.
            n_jobs: How many jobs (threads on different CPU cores) to use
            (where applicable). If -1 uses all available cores. Defaults to -1.
            random_seed: Random seed to ensure results are reproducible.
        """
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.results = {}

        assert growth_phases in [*Experiment.growth_phases,
                                 'all'], \
            "Growth phases must be one of the following: 'lag', 'log', " \
            "'stat', 'all'. Instead got '%s'." % growth_phases

        if growth_phases == 'all':
            growth_phases = Experiment.growth_phases.copy()

        df_16ms = pd.read_csv('data/bacteria_16ms.csv',
                              header=[0, 1, 2, 3],
                              index_col=0)
        df_32ms = pd.read_csv('data/bacteria_32ms.csv',
                              header=[0, 1, 2, 3],
                              index_col=0)

        self.data = {
            '16ms': df_16ms[growth_phases],
            '32ms': df_32ms[growth_phases]
        }

        for df in self.data.values():
            if df.isnull().values.any():
                df.dropna(axis=0, inplace=True)

        self.X = {}
        self.X_pca = {}
        self.y = {}

        self._create_X_y()

        self.cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=20,
                                          random_state=random_seed)

    def _create_X_y(self):
        """Create the X, X_pca and y data sets."""
        if isinstance(Experiment.growth_phases, list):
            for it in Experiment.integration_times:
                dfs = []

                for gp in Experiment.growth_phases:
                    gp_df = self.data[it].T
                    gp_df = gp_df.add_prefix('%s_' % gp)

                    dfs.append(gp_df)

                self.X[it] = pd.concat(dfs, axis=1)
        elif isinstance(Experiment.growth_phases, str):
            for it in Experiment.integration_times:
                self.X[it] = self.data[it].T
                self.X[it].add_prefix('%s_' % Experiment.growth_phases)
        else:
            raise TypeError(
                'Invalid type for parameter growth_phases. Expected a list or'
                ' a string, instead got a %s' % type(Experiment.growth_phases))

        for it in Experiment.integration_times:
            self.y[it] = self.X[it].reset_index()['species']

        self._scale_X()
        self._shuffle_X_y()

        for it in Experiment.integration_times:
            pca = PCA(n_components=0.99, svd_solver='full')
            pca.fit(self.X[it])

            self.X_pca[it] = pca.transform(self.X[it])

    def _scale_X(self):
        """Scale all features into the range [0, 1].

        This is done to improve the run time on SVMs with linear kernels.

        Scaling is done 'globally' as opposed to scaling on a per feature
        (per column) basis since the features are technically all the same
        features. This way relative scaling is retained, which is important and
        affects classification performance.
        """
        for it in Experiment.integration_times:
            X = self.X[it]
            X = (X - X.min()) / (X.max() - X.min())

            self.X[it] = X

    def _shuffle_X_y(self):
        """Shuffle the X and y data sets."""
        for it in Experiment.integration_times:
            self.X[it], self.y[it] = shuffle(self.X[it], self.y[it],
                                             random_state=self.random_seed)

    @timed
    def run(self):
        """Run a series of tests."""
        results = self.results

        for it in Experiment.integration_times:
            print('#' * 80)
            print('Running tests for %s integration time.' % it)
            print('#' * 80)

            results[it] = {}

            results[it]['naive_bayes'] = self.naive_bayes_test(it)
            results[it]['svm'] = self.svm_test(it)
            results[it]['random_forest_stumps'] = \
                self.random_forest_test(it, n_estimators=512, max_depth=1)
            results[it]['random_forest'] = \
                self.random_forest_test(it, n_estimators=512, max_depth=3)
            results[it]['adaboost_stumps'] = \
                self.adaboost_test(it, n_estimators=256, max_depth=1)
            results[it]['adaboost'] = \
                self.adaboost_test(it, n_estimators=256, max_depth=3)

        print('All tests done.')

    def get_results(self, clf, integration_time):
        """Get accuracy scores for both X and X_pca training sets.

        Arguments:
            clf: The classifier to evaluate.
            integration_time: The integration time of the data to use for
                              classification. Must be one of '16ms' or '32ms'.

        Returns: A dictionary containing a list of scores for the 'original'
                 (not transformed) X data set and the pca data set.
        """

        X, X_pca, y = self._get_X_y(integration_time)
        results = {}

        scores = cross_val_score(clf, X, y, cv=self.cv)
        print("Accuracy: %0.2f (+/- %0.2f)" %
              (scores.mean(), scores.std() * 2))
        results['original'] = scores

        scores = cross_val_score(clf, X_pca, y, cv=self.cv)
        print("PCA Accuracy: %0.2f (+/- %0.2f)" %
              (scores.mean(), scores.std() * 2))
        results['pca'] = scores

        return results

    def _get_X_y(self, integration_time):
        """Get the X, X_pca, and y data sets for the given integration time.

        Returns: a three-tuple containing the X, X_pca, and y data sets for the
                 given integration time.
        """
        return (self.X[integration_time], self.X_pca[integration_time],
                self.y[integration_time])

    @timed
    def naive_bayes_test(self, integration_time):
        """Run a classification test using a Naive Bayes classifier.

        Arguments:
            integration_time: The integration time of the data to use for
                              classification. Must be one of '16ms' or '32ms'.

        Returns: A dictionary containing a list of scores for the 'original'
                 (not transformed) X data set and the pca data set.
        """
        status = 'Running Naive Bayes tests.'
        print('*' * len(status))
        print(status)
        print('*' * len(status))

        return self.get_results(GaussianNB(), integration_time)

    @timed
    def svm_test(self, integration_time):
        """Run a classification test using a SVM classifier.

        Also perform grid search to find the best parameters for the SVM.

        Arguments:
            integration_time: The integration time of the data to use for
                              classification. Must be one of '16ms' or '32ms'.

        Returns: A dictionary containing a list of scores for the 'original'
                 (not transformed) X data set and the pca data set.
        """
        status = 'Running SVM tests.'
        print('*' * len(status))
        print(status)
        print('*' * len(status))

        param_grid = {
            'kernel': ['linear', 'rbf'],
            'gamma': [10 ** -n for n in range(10)],
            'C': [10 ** n for n in range(-9, 2)]
        }

        _, X_pca, y = self._get_X_y(integration_time)
        clf = SVC()

        grid_search = GridSearchCV(clf, param_grid, cv=self.cv, iid=True,
                                   verbose=1, n_jobs=self.n_jobs)
        grid_search.fit(X_pca, y)

        print('Best grid search score was %.2f with the following settings: %s'
              % (grid_search.best_score_, grid_search.best_params_))

        return self.get_results(grid_search.best_estimator_, integration_time)

    @timed
    def random_forest_test(self, integration_time, n_estimators, max_depth):
        """Run a classification test using a Random Forest classifier.

        Arguments:
            integration_time: The integration time of the data to use for
                              classification. Must be one of '16ms' or '32ms'.
            n_estimators: How many Decision Trees to use.
            max_depth: The max depth of the Decision Trees.

        Returns: A dictionary containing a list of scores for the 'original'
                 (not transformed) X data set and the pca data set.
        """
        status = 'Running RandomForest tests using %d Decision Trees with a ' \
                 'max depth of %d.' % (n_estimators, max_depth)
        print('*' * len(status))
        print(status)
        print('*' * len(status))

        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     random_state=self.random_seed)

        return self.get_results(clf, integration_time)

    @timed
    def adaboost_test(self, integration_time, n_estimators, max_depth):
        """Run a classification test using the AdaBoost algorithm and Decision
        Trees..

        Arguments:
            integration_time: The integration time of the data to use for
                              classification. Must be one of '16ms' or '32ms'.
            n_estimators: How many Decision Trees to use.
            max_depth: The max depth of the Decision Trees.

        Returns: A dictionary containing a list of scores for the 'original'
                 (not transformed) X data set and the pca data set.
        """
        status = 'Running AdaBoost tests using %d Decision Trees with a ' \
                 'max depth of %d.' % (n_estimators, max_depth)
        print('*' * len(status))
        print(status)
        print('*' * len(status))

        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth),
                                 n_estimators=n_estimators,
                                 random_state=self.random_seed)

        return self.get_results(clf, integration_time)

    def _results_df(self):
        """Create a pandas DataFrame from the results dictionary.

        Returns: A DataFrame containing five columns: integration time,
                 classifier, dataset, mean score, and score standard deviation.
        """
        results_array = []
        results = self.results

        for integration_time in results.keys():
            for classifier in results[integration_time].keys():
                for dataset in results[integration_time][classifier].keys():
                    mean = \
                        results[integration_time][classifier][dataset].mean()
                    std = results[integration_time][classifier][dataset].std()

                    results_array.append([integration_time, classifier,
                                          dataset, mean, std])

        return pd.DataFrame(results_array, columns=['integration_time',
                                                    'classifier', 'dataset',
                                                    'mean_score', 'std_score'])

    def plot_results(self):
        """Plot the results as a grouped bar chart.

        Should be called after run().

        Returns: The matplotlib figure and axes objects.
        """

        df = self._results_df()

        n_rows = 1
        n_cols = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8))
        subplot_i = 0

        for ax, it in zip(axes, ['16ms', '32ms']):
            subplot_i += 1

            # Select data. #
            original = df['dataset'] == 'original'
            pca = df['dataset'] == 'pca'
            integration_time = df['integration_time'] == it

            df_original = df[integration_time & original]
            df_pca = df[integration_time & pca]

            # Create bar plots. #
            width = 0.45
            idx = np.arange(len(df_original))

            original_barplot = ax.bar(x=idx, height=df_original['mean_score'],
                                      width=width,
                                      yerr=df_original['score_std'])

            pca_barplot = ax.bar(x=idx + width, height=df_pca['mean_score'],
                                 width=width, yerr=df_pca['score_std'])

            # Add labels to plot. #
            ax.set_title('Classification Scores for Integration Time of '
                         '%s' % it)
            ax.set_xticks(idx + width / 2)
            ax.set_xticklabels(df['classifier'].unique())
            ax.set_xlabel('Classifier')
            ax.set_ylabel('Classification Score')

            # Attach a text label above each bar displaying its height. #
            for barplot in [original_barplot, pca_barplot]:
                for bar in barplot:
                    height = bar.get_height()
                    ax.text(bar.get_x(), height, '%.2f' % height,
                            ha='left', va='bottom')

            # Add legend to the right of the last plot. #
            if subplot_i == n_rows * n_cols:
                ax.legend((original_barplot[0], pca_barplot[0]),
                          ('None', 'PCA'),
                          title='Transform',
                          bbox_to_anchor=(1, 0.5),
                          fancybox=True,
                          shadow=True)

            ax.autoscale_view()

        fig.suptitle('Classification Scores on Bacteria Fluorescence Spectra')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig, axes

    def top_three(self):
        """Get the top three configurations and their scores.

        Should be called after run().
        """
        df = self._results_df()

        best = df.sort_values(by=['mean_score', 'score_std'],
                              ascending=[False, True])
        best = best.reset_index(drop=True)
        best['mean_score'] = best['mean_score'].map('{:.2f}'.format)
        best['score_std'] = best['score_std'].map('{:.2f}'.format)

        return best.head(3)


class GramnessExperiment(Experiment):
    """Runs a series of classification tests on the bacteria fluorescence
    dataset where the problem is simplified to classifying gramness
    (positive/negative.
    """

    def _create_X_y(self):
        """Create the X, X_pca and y data sets."""
        if isinstance(Experiment.growth_phases, list):
            for it in Experiment.integration_times:
                dfs = []

                for gp in Experiment.growth_phases:
                    gp_df = self.data[it].T
                    gp_df = gp_df.add_prefix('%s_' % gp)

                    dfs.append(gp_df)

                self.X[it] = pd.concat(dfs, axis=1)
        elif isinstance(Experiment.growth_phases, str):
            for it in Experiment.integration_times:
                self.X[it] = self.data[it].T
                self.X[it].add_prefix('%s_' % Experiment.growth_phases)
        else:
            raise TypeError(
                'Invalid type for parameter growth_phases. Expected a list or'
                ' a string, instead got a %s' % type(Experiment.growth_phases))

        for it in Experiment.integration_times:
            self.y[it] = self.X[it].reset_index()['gramness']

        self._scale_X()
        self._shuffle_X_y()

        for it in Experiment.integration_times:
            pca = PCA(n_components=0.99, svd_solver='full')
            pca.fit(self.X[it])

            self.X_pca[it] = pca.transform(self.X[it])


if __name__ == '__main__':
    e = Experiment()
    e.run()
