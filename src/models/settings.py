import numpy as np

from src.models.line import Line


class Settings:

    def __init__(self):
        self.mtx = None
        self.dist = None
        self.aoi_xmid = 0.0
        self.aoi_ymid = 0.0
        self.aoi_upsz = 0.0
        self.aoi_basesz = 0.0
        self.aoi_des = 0.0
        self.aoi_src = None
        self.bird_dst = None
        self.left_glines = Line()
        self.right_glines = Line()
        self.offset = 0

    def find_aoi_src_dst(self, img):
        """
        corners:
           sb ---- sc      db ----- dc
           |        |       |       |
          |          |      |       |
         |            |     |       |
         a ---------- d     a ----- d    
              src              dst
        """
        ys = img.shape[0]
        xs = img.shape[1]
        xmid = xs * self.aoi_xmid - self.aoi_des
        ymid = ys * self.aoi_ymid
        upsz = ys * self.aoi_upsz - self.aoi_des
        basesz = xs * self.aoi_basesz
        a = (xmid - basesz, ys)
        sb = (xmid - upsz, ymid)
        sc = (xmid + upsz, ymid)
        db = (xmid - basesz, 0)
        dc = (xmid + basesz, 0)
        d = (xmid + basesz, ys)
        self.aoi_src = np.float32([[a, sb, sc, d]])
        self.bird_dst = np.float32([[a, db, dc, d]])