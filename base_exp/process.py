import pandas as pd
import numpy as np
import re

def process(df):
    df['full_text'] = df['full_text'].str.replace('Generic_School', 'Generic_school')
    return df