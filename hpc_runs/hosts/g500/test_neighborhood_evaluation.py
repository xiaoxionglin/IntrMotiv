"""Focused metric checks; run with the matching IntrMotiv runtime on PYTHONPATH."""
import numpy as np
from evaluate_neighborhood import metrics


def test_silent_units_are_undefined_not_perfect():
    pose=np.array([[200,200,0],[200,200,10],[900,900,180]],dtype=np.float32)
    result,_=metrics(pose,np.zeros((3,2),np.float32),pose[:2],np.zeros(2,dtype=bool))
    assert result['silent_units']==2
    assert result['per_unit']['positive_recall']==[None,None]
    assert result['per_unit']['spatial_rms_radius']==[None,None]
    assert result['population_active_fraction']==0


def test_heading_wrap_and_negative_counts():
    pose=np.array([[200,200,359],[200,200,1],[200,200,180],[900,900,0]],dtype=np.float32)
    result,_=metrics(pose,np.array([[1],[1],[0],[0]],np.float32),pose[:1],np.ones(1,dtype=bool))
    assert result['per_unit']['positive_count']==[2]
    assert result['per_unit']['negative_count']==[2]
    assert result['per_unit']['positive_recall']==[1.0]
    assert result['per_unit']['false_positive_rate']==[0.0]
