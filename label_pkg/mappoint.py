"class to represent a map point in SLAM results"

__author__ = 'Zherlock'

class Property():
    """Property of a Mappoint
    Attributes:
    x: A int for mappoint x coordinate in the frame
    y: A int for mappoint x coordinate in the frame
    class_id: A int
    class_name: A str
    score: float for maskrcnn inference score
    instance_id: A int
    """

    def __init__(self):
        self.x = -1.0
        self.y = -1.0
        self.class_id = -1
        self.class_name = 'background'
        self.score = 0
        self.instance_id = -1



class MapPoint(object):
    """Mappoint in ORBSLAM
    Attributes: global_id: A int represent the global id of the mappoint when reading from SLAM.
    info: A dictionary for storing information in many mats
    x: A int for mappoint x coordinate in the frame
    y: A int for mappoint x coordinate in the frame
    class_id: A int
    class_name: A str
    score: float for maskrcnn inference score
    instance_id: A int
    fresh: A bool that indicates if this mappoint is merged into a instance
    """

    def __init__(self, global_id):
        self.global_id = global_id
        self.info = {}
        self.class_id = -1
        self.class_name = ''
        self.instance_id = -1
        self.fresh = True