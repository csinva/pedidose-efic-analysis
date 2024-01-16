import pandas as pd
import os
import os.path
from os.path import join
import numpy as np
import imodelsx
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR_INITIAL = '../data/raw/'
DATA_DIR = join(DATA_DIR_INITIAL, '2nd Pass Analysis')


def load_files_dict_single_site():
    files_dict = {}
    files_succ = []
    files_fail = []
    annotators = [x for x in os.listdir(DATA_DIR) if not '.' in x]
    for annotator in annotators:
        print(annotator)
        annotator_dir = join(DATA_DIR, annotator)
        interviews = [x for x in os.listdir(
            annotator_dir) if x.endswith('.xlsx') and not 'INCOMPLETE' in x]
        # files_dict[annotator] = interviews
        for interview in interviews:
            fname = join(annotator_dir, interview)
            df = pd.read_excel(fname)
            key = interview.replace('Interview Analysis.xlsx', '').strip()
            files_dict[key] = df
            COLS = ['Item #', 'Slide #', 'Domain', 'Subcategory']
            for i in range(4):
                assert df.columns[i] == COLS[i]
            assert df.shape[0] == 46
            # print(df.shape)
            # assert df.columns[0] == 'Item #'
            # assert df.columns[1] == 'Slide #'
            # display(df.head())
            # print(df.columns)
            try:
                # , interview + ' ' + df.columns
                assert 'theme' in pd.Series(df.columns).apply(str.lower)
                theme_index = np.where(
                    np.array(list(map(str.lower, df.columns.values))) == 'theme')[0][0]
                for i in range(theme_index + 1, len(df.columns), 2):
                    assert '#' in df.columns[i]
                assert df.shape[0] == 46
                files_succ.append(key)
            except:
                files_fail.append(key)
    print('success:', len(files_succ))
    print('fail:', files_fail)
    return files_dict


def get_data_for_question_single_site(question_num: int, qs, responses_df, themes_df):
    question = qs.iloc[question_num]
    responses = responses_df.iloc[question_num]
    theme_row = themes_df.iloc[question_num].values
    theme_dict = {theme_row[i]: theme_row[i + 1]
                  for i in range(0, len(theme_row), 2)}

    for k in list(theme_dict.keys()):
        # remove nan/whitespace from theme_dict
        if type(k) != str or not k.strip():
            del theme_dict[k]

    return question, responses, theme_dict
