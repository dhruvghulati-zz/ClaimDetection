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

# TODO - need to create a hyperparameter for the actual normalised indices for the sigmoid

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
            {'a': 3,'b': 98,'c':1711},
            {'a': 121,'b': 12121,'c': 12989121},
        ]
    },
    'open_cost_2':{
        'cost_matrix': [
            {'a': 123,'b': 1312,'c': 1231},
            {'a': 1011,'b': 1911,'c':911},
            {'a': 1433,'b': 19829,'c': 1132},
        ]
    }
}

def apply_normalizations(costs):
    """Add a 'normalised_matrix' next to each 'cost_matrix' in the values of costs"""

    # TODO - make this have parameters that can be adjusted
    def sigmoid(v):
        return 1 / (1 + np.exp(-v))

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
                    values_exp.append(float(1e10))
        return sum(values_exp)

    def sum_sq(a,lst):
        values_squared = [(a*v)**2 for v in lst if v is not None]
        return sum(values_squared)

    def normalize(v, least, most):
        return 1.0 if least == most else float(v - least) / (most - least)

    def normalize_sums(v, sum):
        return float(v) / sum

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


    for name, value in costs.items():
        if int((name.split("_")[-1]))>1:
            value['normalised_matrix'] = normalize_dicts_local(value['cost_matrix'])
            value['normalised_matrix_sum'] = normalize_dicts_local_sum(value['cost_matrix'])
            value['normalised_matrix_sumSquared'] = normalize_dicts_local_sq(value['cost_matrix'])
            value['normalised_matrix_sumExp'] = normalize_dicts_local_exp(value['cost_matrix'])
            value['normalised_matrix_sigmoid'] = normalize_dicts_local_sigmoid(value['cost_matrix'])


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