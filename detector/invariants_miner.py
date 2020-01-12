#!/usr/bin/env python
# -*-coding: utf-8 -*-

import json
from itertools import combinations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class InvariantsMiner(object):
    def __init__(self, percentage=0.98, epsilon=0.5, longest_invarant=None, scale_list=[1, 2, 3]):
        """ The Invariants Mining model for anomaly detection

        Attributes
        ----------
            percentage: float, percentage of samples satisfying the condition that |X_j * V_i| < epsilon
            epsilon: float, the threshold for estimating the invariant space
            longest_invarant: int, the specified maximal length of invariant, default to None. Stop
                searching when the invariant length is larger than longest_invarant.
            scale_list: list, the list used to scale the theta of float into integer
            invariants_dict: dict, dictionary of invariants where key is the selected columns
                and value is the weights the of invariant
        """
        self.percentage = percentage
        self.epsilon = epsilon
        self.longest_invarant = longest_invarant
        self.scale_list = scale_list
        self.invariants_dict = None
        self.invariants_count_dict = None
        self.events = None

    def fit(self, X, events):
        """
        Arguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """
        print('====== Model summary ======')
        invar_dim = self._estimate_invarant_space(X)
        self._invariants_search(X, invar_dim, events)

    def predict(self, X):
        """ Predict anomalies with mined invariants

        Arguments
        ---------
            X: the input event count matrix

        Returns
        -------
            y_pred: ndarray, the predicted label vector of shape (num_instances,)
        """

        # print("invariants_dict: ", self.invariants_dict)
        # np.set_printoptions(threshold=np.inf)
        y_sum = np.zeros(X.shape[0])
        for cols, theta in self.invariants_dict.items():
            y_sum += np.fabs(np.dot(X[:, cols], np.array(theta)))
        # print("y_sum", y_sum)
        y_pred = (y_sum > 1e-6).astype(int)
        for idx, value in enumerate(y_pred):
            if value > 0:
                print(idx)
        return y_pred

    def predict_new(self, X, orig_data):
        """ Predict anomalies with mined invariants

        Arguments
        ---------
            X: the input event count matrix

        Returns
        -------
            y_pred: ndarray, the predicted label vector of shape (num_instances,)
        """

        # print("invariants_dict: ", self.invariants_dict)
        # np.set_printoptions(threshold=np.inf)
        invariants_count_dict = dict()
        for cols, theta in self.invariants_dict.items():
            count = 0
            for i in range(X.shape[0]):
                if np.sum([a ** 2 for a in X[i, cols]]) == 0:
                    continue
                dv = np.dot(X[i, cols], np.array(theta))
                if dv == 0:
                    count += 1
            invariants_count_dict[cols] = count
        self.invariants_count_dict = invariants_count_dict

        uniq_dict = dict()
        y_sum = np.zeros(X.shape[0])
        for cols, theta in self.invariants_dict.items():
            y_sum += np.fabs(np.dot(X[:, cols], np.array(theta)))
        # print("y_sum", y_sum)
        y_pred = (y_sum > 1e-6).astype(int)
        total = 0
        for idx, value in enumerate(y_pred):
            if value > 0:
                event_repr = ",".join(orig_data.iloc[idx][1])
                if event_repr in uniq_dict:
                    uniq_dict[event_repr] += 1
                else:
                    uniq_dict[event_repr] = 1
                # print(idx, orig_data.iloc[idx][1])
                total += 1
        print("uniq_dict len: ", len(uniq_dict))
        # print(uniq_dict)
        print("Total unmatched invalid: ", total)
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1 = self.metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1

    def _estimate_invarant_space(self, X):
        """ Estimate the dimension of invariant space using SVD decomposition

        Arguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
            percentage: float, percentage of samples satisfying the condition that |X_j * V_i| < epsilon
            epsilon: float, the threshold for estimating the invariant space

        Returns
        -------
            r: the dimension of invariant space
        """
        covariance_matrix = np.dot(X.T, X)
        U, sigma, V = np.linalg.svd(covariance_matrix)  # SVD decomposition
        # Start from the right most column of matrix V, sigular values are in ascending order
        num_instances, num_events = X.shape
        r = 0
        for i in range(num_events - 1, -1, -1):
            zero_count = sum(abs(np.dot(X, U[:, i])) < self.epsilon)
            if zero_count / float(num_instances) < self.percentage:
                break
            # print('current invariant: ', U[:, i])
            r += 1
        print('Invariant space dimension: {}'.format(r))

        return r

    def _invariants_search(self, X, r, events):
        """ Mine invariant relationships from X

        Arguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
            r: the dimension of invariant space
        """

        num_instances, num_events = X.shape
        invariants_dict = dict()  # save the mined Invariants(value) and its corresponding columns(key)
        search_space = []  # only invariant candidates in this list are valid

        # invariant of only one column (all zero columns)
        init_cols = sorted([[item] for item in range(num_events)])
        # print(init_cols)
        for col in init_cols:
            search_space.append(col)
        init_col_list = init_cols[:]
        for col in init_cols:
            if np.count_nonzero(X[:, col]) == 0:
                invariants_dict[tuple(col)] = [1]
                search_space.remove(col)
                init_col_list.remove(col)
        # print(search_space)

        item_list = init_col_list
        length = 2
        FLAG_break_loop = False
        # check invariant of more columns
        while len(item_list) != 0:
            if length >= 4:
                break
            if self.longest_invarant and len(item_list[0]) >= self.longest_invarant:
                break
            # item_list组合出长度为length的新集合
            joined_item_list = self._join_set(item_list, length)  # generate new invariant candidates
            for items in joined_item_list:
                # 若items的所有排列组合都在search_space中，则将其加入search_space中作为候选不变量
                if self._check_candi_valid(items, length, search_space):
                    search_space.append(items)
                    # print("search_space append: ", items)
                    # print("search_space: ", search_space)
            item_list = []
            print("current join_item_list length: ", len(joined_item_list))
            for item in joined_item_list:
                if tuple(item) in invariants_dict:
                    continue
                if item not in search_space:
                    continue
                if not self._check_candi_valid(tuple(item), length, search_space) and length > 2:
                    search_space.remove(item)
                    continue  # an item must be superset of all other subitems in searchSpace, else skip
                # 判断
                validity, scaled_theta = self._check_invar_validity(X, item)
                if validity:
                    self._prune(invariants_dict.keys(), set(item), search_space)
                    invariants_dict[tuple(item)] = scaled_theta.tolist()
                    search_space.remove(item)
                    print("Current invariant dict: ", invariants_dict)
                    # print("Current search_space: ", search_space)
                else:
                    item_list.append(item)
                if len(invariants_dict) >= r:
                    FLAG_break_loop = True
                    break
            if FLAG_break_loop:
                break
            length += 1
        print('Mined {} invariants: {}\n'.format(len(invariants_dict), invariants_dict))
        self.invariants_dict = invariants_dict
        self.events = events
        self.get_actual_event_repr()

    def load_invariants(self, file_name, events):
        with open(file_name) as f:
            contents = json.load(f)
            template_map = contents['EventTemplate']
            print(template_map)
            invariant_dict = dict()
            event_list = list(events)
            for invariant in contents['Invariants']:
                event_keys = [idx for idx in invariant if idx in event_list]
                print(event_keys)
                cal_key = tuple([event_list.index(idx) for idx in event_keys])
                invariant_dict[cal_key] = [invariant[idx] for idx in event_keys]
            print("Current invarints: ", invariant_dict)
            self.invariants_dict = invariant_dict
            self.events = events

    def get_actual_event_repr(self):
        print("Invariant dict: ")
        if isinstance(self.invariants_dict, dict):
            for key, value in self.invariants_dict.items():
                actual_key = tuple([self.events[idx] for idx in key])
                print(actual_key, value)
        print("Invariant count: ")
        if isinstance(self.invariants_count_dict, dict):
            for key, value in self.invariants_count_dict.items():
                actual_key = tuple([self.events[idx] for idx in key])
                print(actual_key, value)
        pass

    def _compute_eigenvector(self, X):
        """ calculate the smallest eigenvalue and corresponding eigenvector (theta in the paper)
            for a given sub_matrix

        Arguments
        ---------
            X: the event count matrix (each row is a log sequence vector, each column represents an event)

        Returns
        -------
            min_vec: the eigenvector of corresponding minimum eigen value
            FLAG_contain_zero: whether the min_vec contains zero (very small value)
        """

        FLAG_contain_zero = False
        count_zero = 0
        dot_result = np.dot(X.T, X)
        U, S, V = np.linalg.svd(dot_result)
        min_vec = U[:, -1]
        count_zero = sum(np.fabs(min_vec) < 1e-6)
        if count_zero != 0:
            FLAG_contain_zero = True
        return min_vec, FLAG_contain_zero

    def _check_invar_validity(self, X, selected_columns):
        """ scale the eigenvector of float number into integer, and check whether the scaled number is valid

        Arguments
        ---------
            X: the event count matrix (each row is a log sequence vector, each column represents an event)
            selected_columns: select columns from all column list

        Returns
        -------
            validity: whether the selected columns is valid
            scaled_theta: the scaled theta vector
        """

        sub_matrix = X[:, selected_columns]
        inst_num = X.shape[0]
        validity = False
        min_theta, FLAG_contain_zero = self._compute_eigenvector(sub_matrix)
        abs_min_theta = [np.fabs(it) for it in min_theta]
        if FLAG_contain_zero:
            return validity, []
        else:
            for i in self.scale_list:
                min_index = np.argmin(abs_min_theta)
                scale = float(i) / min_theta[min_index]
                scaled_theta = np.array([round(item * scale) for item in min_theta])
                scaled_theta[min_index] = i
                scaled_theta = scaled_theta.T
                if 0 in np.fabs(scaled_theta):
                    continue
                # 计算每一行的点积
                dot_submat_theta = np.dot(sub_matrix, scaled_theta)
                count_zero = 0
                for j in dot_submat_theta:
                    if np.fabs(j) < 1e-8:
                        count_zero += 1
                if count_zero >= self.percentage * inst_num:
                    validity = True
                    print('A valid invariant is found: ', scaled_theta, selected_columns)
                    break
            return validity, scaled_theta

    def _prune(self, valid_cols, new_item_set, search_space):
        """ prune invalid combination of columns

        Arguments
        ---------
            valid_cols: existing valid column list
            new_item_set: item set to be merged
            search_space: the search space that stores possible candidates

        """

        if len(valid_cols) == 0:
            return
        for se in valid_cols:
            intersection = set(se) & new_item_set
            if len(intersection) == 0:
                continue
            union = set(se) | new_item_set
            for item in list(intersection):
                diff = sorted(list(union - set([item])))
                if diff in search_space:
                    search_space.remove(diff)

    def _join_set(self, item_list, length):
        """ Join a set with itself and returns the n-element (length) itemsets

        Arguments
        ---------
            item_list: current list of columns
            length: generate new items of length

        Returns
        -------
            return_list: list of items of length-element
        """

        set_len = len(item_list)
        return_list = []
        for i in range(set_len):
            for j in range(i + 1, set_len):
                i_set = set(item_list[i])
                j_set = set(item_list[j])
                if len(i_set.union(j_set)) == length:
                    joined = sorted(list(i_set.union(j_set)))
                    if joined not in return_list:
                        return_list.append(joined)
        return_list = sorted(return_list)
        return return_list

    def _check_candi_valid(self, item, length, search_space):
        """ check whether an item's subitems are in searchspace

        Arguments
        ---------
            item: item to be checked
            length: the length of item
            search_space: the search space that stores possible candidates

        Returns
        -------
            True or False
        """

        for subItem in combinations(item, length - 1):
            if sorted(list(subItem)) not in search_space:
                return False
        return True

    def metrics(self, y_pred, y_true):
        """ Calucate evaluation metrics for precision, recall, and f1.

        Arguments
        ---------
            y_pred: ndarry, the predicted result list
            y_true: ndarray, the ground truth label list

        Returns
        -------
            precision: float, precision value
            recall: float, recall value
            f1: float, f1 measure value
        """
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        return precision, recall, f1


if __name__ == '__main__':
    item_list = [[1, 2], [2], [3], [4], [5]]
    print(InvariantsMiner()._join_set(item_list, 3))
