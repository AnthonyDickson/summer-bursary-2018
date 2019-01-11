import os
from time import time

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle


def timed(func):
    """A simple decorator that prints the elapsed time of the function call."""
    def wrapper(*args, **kwargs):
        start = time()

        func(*args, **kwargs)

        print('Elapsed time: %.2fs.' % (time() - start))

    return wrapper


class Experiment:
    growth_phases = ['lag', 'log', 'stat']
    integration_times = ['16ms', '32ms']

    def __init__(self, growth_phases='all', n_jobs=-1, random_seed=42):
        """Create an experiment that gets classification scores with various setups.

        Arguments:
            growth_phases: Which growth_phases of the dataframe to use. Must be one of 'lag', 'log', 'stat', or 'all'. Defaults to 'all'.
            n_jobs: How many jobs (threads on different CPU cores) to use (where applicable). If -1 uses all available cores. Defaults to -1.
            random_seed: Random seed to ensure results are reproducible.
        """
        self.n_jobs = n_jobs
        self.random_seed = random_seed

        expected_working_dir = 'summer-bursary-2018/bacteria'

        assert os.getcwd().endswith(
            expected_working_dir), "Expected to be in working directory ending with '%s', working directory is instead '%s'" % (
        expected_working_dir, os.getcwd())

        assert growth_phases in [*Experiment.growth_phases,
                                 'all'], "Growth phases must be one of the following: 'lag', 'log', 'stat', 'all'. It was instead '%s'." % growth_phases

        if growth_phases == 'all':
            growth_phases = Experiment.growth_phases.copy()

        df_16ms = pd.read_csv('data/bacteria.csv', header=[0, 1, 2], index_col=0)
        df_32ms = pd.read_csv('data/bacteria.csv', header=[0, 1, 2], index_col=0)

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
                'Invalid type for parameter growth_phases. Expected a list or a string, instead got a %s' % type(
                    Experiment.growth_phases))

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
    def run_all(self):
        """Run all tests."""
        results = {}

        for it in Experiment.integration_times:
            print('#' * 80)
            print('Running tests for %s integration time.' % it)
            print('#' * 80)

            X, X_pca, y = self.X[it], self.X_pca[it], self.y[it]
            results[it] = {}

            results[it]['nb'] = self.naive_bayes_stuff(X, X_pca, y)
            results[it]['svm'] = self.svm_stuff(X, X_pca, y)
            results[it]['rf_stumps'] = \
                self.random_forest_stuff(X, X_pca, y,
                                         n_estimators=512, max_depth=1)
            results[it]['rf'] = \
                self.random_forest_stuff(X, X_pca, y,
                                         n_estimators=128, max_depth=3)
            results[it]['ada'] = \
                self.adaboost_stuff(X, X_pca, y, n_estimators=256, max_depth=1)
            results[it]['ada'] = \
                self.adaboost_stuff(X, X_pca, y, n_estimators=64, max_depth=3)

        print('All tests done.')

    def get_results(self, clf, X, X_pca, y):
        """Get accuracy scores for both X and X_pca training sets."""
        results = []

        scores = cross_val_score(clf, X, y, cv=self.cv)
        print("Accuracy: %0.2f (+/- %0.2f)" %
              (scores.mean(), scores.std() * 2))
        results.append((scores.mean(), scores.std() * 2))

        scores = cross_val_score(clf, X_pca, y, cv=self.cv)
        print("PCA Accuracy: %0.2f (+/- %0.2f)" %
              (scores.mean(), scores.std() * 2))
        results.append((scores.mean(), scores.std() * 2))

        return results

    @timed
    def naive_bayes_stuff(self, X, X_pca, y):
        """Do naive Bayes stuff."""
        status = 'Running Naive Bayes tests.'
        print('*' * len(status))
        print(status)
        print('*' * len(status))

        return self.get_results(GaussianNB(), X, X_pca, y)

    @timed
    def svm_stuff(self, X, X_pca, y):
        """Do SVM stuff."""
        status = 'Running SVM tests.'
        print('*' * len(status))
        print(status)
        print('*' * len(status))

        param_grid = {
            'kernel': ['linear', 'rbf'],
            'gamma': [10 ** -n for n in range(10)],
            'C': [10 ** n for n in range(-9, 2)]
        }

        clf = SVC()

        grid_search = GridSearchCV(clf, param_grid, cv=self.cv, iid=True,
                                   verbose=1, n_jobs=self.n_jobs)
        grid_search.fit(X_pca, y)

        print('Best grid search score was %.2f with the following settings: %s'
              % (grid_search.best_score_, grid_search.best_params_))

        return self.get_results(grid_search.best_estimator_, X, X_pca, y)

    @timed
    def random_forest_stuff(self, X, X_pca, y, n_estimators, max_depth):
        """Do random forest stuff.

        Arguments:
             n_estimators: How many Decision Trees to use.
             max_depth: The max depth of the Decision Trees.
        """
        status = 'Running RandomForest tests using %d Decision Trees with a ' \
                 'max depth of %d.' % (n_estimators, max_depth)
        print('*' * len(status))
        print(status)
        print('*' * len(status))

        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     random_state=self.random_seed)

        return self.get_results(clf, X, X_pca, y)

    @timed
    def adaboost_stuff(self, X, X_pca, y, n_estimators, max_depth):
        """Do AdaBoost stuff.

        Arguments:
             n_estimators: How many Decision Trees to use.
             max_depth: The max depth of the Decision Trees.
        """
        status = 'Running AdaBoost tests using %d Decision Trees with a ' \
                 'max depth of %d.' % (n_estimators, max_depth)
        print('*' * len(status))
        print(status)
        print('*' * len(status))

        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth),
                                 n_estimators=n_estimators,
                                 random_state=self.random_seed)

        return self.get_results(clf, X, X_pca, y)


if __name__ == '__main__':
    e = Experiment()
    e.run_all()
