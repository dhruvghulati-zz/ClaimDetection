import numpy as np

values = [u'/location/statistical_region/population', u'/location/statistical_region/diesel_price_liter', u'/location/statistical_region/gni_per_capita_in_ppp_dollars']

values = np.array([str(item).split('/')[3] for item in values])

print values

