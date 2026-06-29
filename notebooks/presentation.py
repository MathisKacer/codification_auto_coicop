# %%
import sys
sys.path.append("/home/onyxia/work/codification_auto_coicop")
from data.load_data import charger_base
import pandas as pd

# %%
df = charger_base()

# %%
print(df.columns)

# %%
print(df.head)
# %%
