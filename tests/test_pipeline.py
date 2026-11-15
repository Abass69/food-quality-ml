import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from food_quality import load_data, grades, FEATURES, validate_input

class PipelineTests(unittest.TestCase):
    def test_grade_boundaries(self):
        np.testing.assert_array_equal(grades([3,5,6,7,9]),[0,0,1,2,2])
    def test_data_and_split(self):
        df,g,a=load_data()
        self.assertEqual(a['raw_rows'],6497)
        self.assertEqual(a['missing_values'],0)
        self.assertEqual(len(df.drop_duplicates(FEATURES+['quality'])),len(df))
        for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=42).split(df,grades(df.quality),g):
            self.assertFalse(set(g[tr])&set(g[te]))
    def test_reject_invalid_inputs(self):
        df,_,_=load_data(); X=df[FEATURES].head(1).copy()
        validate_input(X)
        with self.assertRaises(ValueError): validate_input(X.drop(columns=['pH']))
        X.loc[0,'pH']=np.nan
        with self.assertRaises(ValueError): validate_input(X)
        X.loc[0,'pH']=3; X.loc[0,'wine_type']=2
        with self.assertRaises(ValueError): validate_input(X)

if __name__=='__main__': unittest.main()
