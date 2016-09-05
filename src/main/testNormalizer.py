import numpy as np
import math
from scipy.stats.mstats import zscore
from scipy.special import expit

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
            {'a': 0.005,'b': 0.0543,'c':911},
            {'a': 1433,'b': 19829,'c': 1132},
        ]
    }
}

def apply_normalizations(costs):
    """Add a 'normalised_matrix' next to each 'cost_matrix' in the values of costs"""

    # TODO - make this have parameters that can be adjusted
    def sigmoid(b,m,v):
        return expit(b + m*v)*2 - 1
        # ((expit(b+m*v) / (1 + expit(b+m*v)))*2)-1

    def normalize_dicts_local_sigmoid(bias, slope,lst):
        return [{key: sigmoid(bias, slope,val) for key,val in dic.iteritems()} for dic in lst]


    for name, value in costs.items():
        value['normalised_matrix_sigmoid'] = normalize_dicts_local_sigmoid(2,0.5,value['cost_matrix'])


apply_normalizations(actualarray)

print actualarray


# def sigmoid(b,m,v):
#     return expit(b + m*v)*2 - 1
#     # ((expit(b+m*v) / (1 + expit(b+m*v)))*2)-1
#
# def normalize_dicts_local_sigmoid(bias, slope,lst):
#     return [{key: sigmoid(bias, slope,val) for key,val in dic.iteritems()} for dic in lst]
#
#
# for name, value in actualarray.items():
#     for i,array in enumerate(value['cost_matrix']):
#         for key, value in array.items():
#             print key
        # value['normalised_matrix_sigmoid'] = normalize_dicts_local_sigmoid(0,1,value['cost_matrix'])


# print actualarray


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