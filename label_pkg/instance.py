"class to represent an instance in SLAM and rcnn results"

__author__ = 'Zherlock'

class Instance():
    """abstraction of the instance
    Attributes: mps: set of int which are map point ids which belong to this instance
    cls: A str for instance classname
    id: A int for instance id
    mat_set: A set of sub mats which belong to this instance
    """
    def __init__(self):
        self.mps = set()
        self.cls = ''
        self.id = -1
        self.mat_set = set()