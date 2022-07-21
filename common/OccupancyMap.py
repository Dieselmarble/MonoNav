import numpy as np

class OccupancyMap:
    def __init__(self, seg, floorID):
        self.seg = seg
        self.floorID = floorID

    def convert2Occu(self):
        segMap = self.seg
        # 0 stands for obstacle and 1 stands for available zone
        occuMap = np.ones(segMap.shape, dtype=np.uint8)
        occuMap[segMap!=self.floorID] = 0
        return occuMap
