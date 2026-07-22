
import json
aligned_items = {int(k): v for k, v in json.load(open("/Users/tanmoy/research/Credal_Sets/Random_Sets/rsuq/ensemble_lp_chaosnli.json")).items()}

print (aligned_items)