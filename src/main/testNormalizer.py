import numpy as np
import math
from scipy.stats.mstats import zscore


# for model,array in actualarray.items():
#     print array['cost_matrix']
#
# values = sum([[item["value"]] + item["openValues"].values()
#               for item in array], [])
#
# print values
# v_min, v_max = min(values), max(values)
# output = [f(v_min, v_max, item) for item in array]
# print output

array = [
    {
        'value': 21,
        'openValues': {
            'a': 24,
            'b': 56,
            'c': 78
        }
    },
    {
        'value': 12,
        'openValues': {
            'a': 98,
            'b': 3
        }
    },
    {
        'value': 900,
        'openValues': {
            'a': 7811,
            'b': 171,
            'c': 11211,
            'd': 4231
        }
    }
]


# def normalize(v0, v1, t):
#     if v1-v0==0:
#         return float(1)
#     else:
#         return float(t - v0) / float(v1 - v0)


# def f(v0, v1, values):
#     return [normalize(v0, v1, item) for item in values]

actualarray = {
    'open_cost_1':{
        'cost_matrix': [
            {'a': 24,'b': 56,'c': 78},
            {'a': 3,'b': 98},
            {'a': 121,'b': 12121,'c': 12989121,'d':16171},
        ]
    },
    'open_cost_2':{
        'cost_matrix': [
            {'a': 123,'b': 1312,'c': 1231},
            {'a': 1011,'b': 1911},
            {'a': 1433,'b': 19829,'c': 1132,'d':1791},
        ]
    },
    'single_open_cost_1':{
        'cost_matrix': [
            34,
            56,
            98
        ]
    },
    'single_open_cost_2':{
        'cost_matrix': [
            1811,
            1211,
            1267
        ]
    }
}

def apply_normalizations(costs):
    """Add a 'normalised_matrix' next to each 'cost_matrix' in the values of costs"""

    '''TODO - split into global and local normalisations to save computation'''

    def sigmoid(v):
        return 1 / (1 + np.exp(-v))

    def void(lst):
        print

    def min_max(lst):
        values = [v for v in lst if v is not None]
        return min(values), max(values)

    def sum_reg(a,lst):
        values = [a*v for v in lst if v is not None]
        return sum(values)

    def sum_exp(a,lst):
        values_exp = []
        for v in lst:
            if v is not None:
                try:
                    values_exp.append(math.exp(a*v))
                except OverflowError:
                    values_exp.append(float('inf'))
        return sum(values_exp)

    def sum_sq(a,lst):
        values_squared = [a*v^2 for v in lst if v is not None]
        return sum(values_squared)

    def mean_reg(lst):
        values = [v for v in lst if v is not None]
        return np.mean(values)

    def standard_dev(lst):
        values = [v for v in lst if v is not None]
        return np.std(values)

    def normalize(v, least, most):
        return 1.0 if least == most else float(v - least) / (most - least)

    def normalize_sums(v, sum):
        return float(v) / sum

    def normalize_zscore(v, mean,std):
        return float(v) - mean / std

    def normalize_nums_global(lst):
        span = min_max(lst)
        return [normalize(val, *span) for val in lst]

    def normalize_nums_global_sum(lst):
        sum = sum_reg(1,lst)
        return [normalize_sums(val, sum) for val in lst]

    def normalize_nums_global_exp(lst):
        sumexp = sum_exp(1,lst)
        return [normalize_sums(val, sumexp) for val in lst]

    def normalize_nums_global_sq(lst):
        sumsq = sum_sq(1,lst)
        return [normalize_sums(val, sumsq) for val in lst]

    def normalize_nums_global_sigmoid(lst):
        return [sigmoid(val) for val in lst]

    def zscore_array(lst):
        return zscore(lst).tolist()

    def normalize_dicts_global(lst):
        keys = set.union(*(set(dic.iterkeys()) for dic in lst))
        spans = {key:min_max(dic.get(key) for dic in lst) for key in keys}
        return [
           {key: normalize(val, *spans[key]) for key, val in dic.iteritems()}
           for dic in lst
        ]

    def normalize_dicts_global_exp(lst):
        keys = set.union(*(set(dic.iterkeys()) for dic in lst))
        exps = {key:sum_exp(1,(dic.get(key) for dic in lst)) for key in keys}
        return [
           {key: normalize_sums(val, exps[key]) for key, val in dic.iteritems()}
           for dic in lst
        ]

    def normalize_dicts_global_sq(lst):
        keys = set.union(*(set(dic.iterkeys()) for dic in lst))
        squares = {key:sum_sq(1,(dic.get(key) for dic in lst)) for key in keys}
        return [
           {key: normalize_sums(val, squares[key]) for key, val in dic.iteritems()}
           for dic in lst
        ]

    def normalize_dicts_global_sums(lst):
        keys = set.union(*(set(dic.iterkeys()) for dic in lst))
        sums = {key:sum_reg(1,(dic.get(key) for dic in lst)) for key in keys}
        return [
           {key: normalize_sums(val, sums[key]) for key, val in dic.iteritems()}
           for dic in lst
        ]

    def normalize_dicts_global_sigmoid(lst):
        return [
           {key: sigmoid(val) for key, val in dic.iteritems()}
           for dic in lst
        ]

    def normalize_dicts_global_zscore(lst):
        keys = set.union(*(set(dic.iterkeys()) for dic in lst))
        means = {key:mean_reg(dic.get(key) for dic in lst) for key in keys}
        stds = {key:standard_dev(dic.get(key) for dic in lst) for key in keys}
        return [
           {key: normalize_zscore(val, means[key],stds[key]) for key, val in dic.iteritems()}
            for dic in lst
        ]

    def normalize_dicts_local(lst):
        spans = [min_max(dic.values()) for dic in lst]
        return [{key: normalize(val,*span) for key,val in dic.iteritems()} for dic,span in zip(lst,spans)]

    def normalize_dicts_local_sum(lst):
        sums = [sum_reg(1,dic.values()) for dic in lst]
        return [{key: normalize_sums(val,tempsum) for key,val in dic.iteritems()} for dic,tempsum in zip(lst,sums)]

    def normalize_dicts_local_exp(lst):
        sums_exp = [sum_exp(1,dic.values()) for dic in lst]
        return [{key: normalize_sums(val,tempsum) for key,val in dic.iteritems()} for dic,tempsum in zip(lst,sums_exp)]

    def normalize_dicts_local_sq(lst):
        sums_sq = [sum_sq(1,dic.values()) for dic in lst]
        return [{key: normalize_sums(val,tempsum) for key,val in dic.iteritems()} for dic,tempsum in zip(lst,sums_sq)]

    def normalize_dicts_local_sigmoid(lst):
        return [{key: sigmoid(val) for key,val in dic.iteritems()} for dic in lst]

    def multiply_dicts_n(lst):
        return [{key: (val)*len(lst) for key,val in dic.iteritems()} for dic in lst]

    def multiply_nums_n(lst):
        return [val*len(lst) for val in lst]

    def normalize_dicts_local_zscore(lst):
        means = [mean_reg(dic.values()) for dic in lst]
        stds = [standard_dev(dic.values()) for dic in lst]
        return [{key: normalize_zscore(val,mean,std) for key,val in dic.iteritems()} for dic,mean,std in zip(lst,means,stds)]

    for name, value in costs.items():
        # These are forms of global normalisation
        norm = normalize_nums_global if name.startswith('single') else normalize_dicts_global
        norm_sum = normalize_nums_global_sum if name.startswith('single') else normalize_dicts_global_sums
        norm_sq = normalize_nums_global_sq if name.startswith('single') else normalize_dicts_global_sq
        norm_exp = normalize_nums_global_exp if name.startswith('single') else normalize_dicts_global_exp
        norm_sigmoid = normalize_nums_global_sigmoid if name.startswith('single') else normalize_dicts_global_sigmoid
        norm_zscore = zscore_array if name.startswith('single') else normalize_dicts_global_zscore
        multiply_n = multiply_nums_n if name.startswith('single') else multiply_dicts_n
        value['globally_normalised_matrix'] = norm(value['cost_matrix'])
        value['globally_normalised_matrix_sum'] = norm_sum(value['cost_matrix'])
        value['globally_normalised_matrix_sumExp'] = norm_exp(value['cost_matrix'])
        value['globally_normalised_matrix_sumSquared'] = norm_sq(value['cost_matrix'])
        value['globally_normalised_matrix_sigmoid'] = norm_sigmoid(value['cost_matrix'])
        value['globally_normalised_matrix_zscore'] = norm_zscore(value['cost_matrix'])
        # Multiply by N as a different example, makes sense to do this even if single cost
        value['cost_matrix_n'] = multiply_n(value['cost_matrix'])
        # Only makes sense to normalise row-wise in certain cases
        if not name.startswith('single'):
            value['locally_normalised_matrix'] = normalize_dicts_local(value['cost_matrix'])
            value['locally_normalised_matrix_sum'] = normalize_dicts_local_sum(value['cost_matrix'])
            value['locally_normalised_matrix_sumSquared'] = normalize_dicts_local_sq(value['cost_matrix'])
            value['locally_normalised_matrix_sumExp'] = normalize_dicts_local_exp(value['cost_matrix'])
            value['locally_normalised_matrix_sigmoid'] = normalize_dicts_local_sigmoid(value['cost_matrix'])
            value['locally_normalised_matrix_zscore'] = normalize_dicts_local_zscore(value['cost_matrix'])

apply_normalizations(actualarray)

print actualarray




# dict_values= {}
# array_values = {}
#
# for outer_key,dict in actualarray.items():
#     if outer_key.startswith("single"):
#         dict_values[outer_key]= {}
#         for inner_dict in dict['cost_matrix']:
#             for key,value in inner_dict.items():
#                 if key not in dict_values[outer_key]:
#                     dict_values[outer_key][key]= []
#                 dict_values[outer_key][key].append(value)
#     else:
#         array_values[outer_key]= []
#         for value in dict['cost_matrix']:
#             array_values[outer_key].append(value)
#             # array_values[i].append(value)

# print array_values
# print dict_values


# for model,values in array_values.items():
#     v_min, v_max = min(values), max(values)
#     actualarray[model]['normalised_matrix'] = [normalize(v_min, v_max, item) for item in values]
#
#
# for outer_key,main_dict in actualarray.items():
#     if outer_key.startswith("single"):
#         actualarray[outer_key]['normalised_matrix'] = []
#         array_dict= dict_values[outer_key]
#         for dict in main_dict['cost_matrix']:
#             temp_dict = {}
#             for key,value in dict.items():
#                 v_min, v_max = min(array_dict[key]), max(array_dict[key])
#                 temp_dict[key]=normalize(v_min, v_max, value)
#             actualarray[outer_key]['normalised_matrix'].append(temp_dict)
#
# print actualarray


# for model,dict in dict_values.iteritems():
#     normalizedarray[model] = {'cost_matrix':[]}
#     # print dict
#     for key,values in dict.items():
#         v_min, v_max = min(values), max(values)
#         # print values
#         empty_dict = {}
#         for value in values:
#             empty_dict[key] = normalize(v_min, v_max, value)
#             print empty_dict
            # normalizedarray[model]['cost_matrix'].append(empty_dict)

# print normalizedarray

# values = sum([, [])
# print values
# v_min, v_max = min(values), max(values)
# output = [f(v_min, v_max, item) for item in array]
# print output