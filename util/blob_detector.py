import cv2
import numpy as np


class BlobDetector:
    def __init__(self,
                 area_bounds: tuple[float, float] = None,
                 circularity_bounds: tuple[float, float] = None,
                 convexity_bounds: tuple[float, float] = None,
                 inertia_bounds: tuple[float, float] = None,
                 ):
        self.params = cv2.SimpleBlobDetector.Params()
        self.params.filterByArea = area_bounds is not None
        self.params.minArea = area_bounds[0] if self.params.filterByArea else None
        self.params.maxArea = area_bounds[1] if self.params.filterByArea else None

        self.params.filterByCircularity = circularity_bounds is not None
        self.params.minCircularity = circularity_bounds[0] if self.params.filterByCircularity else None
        self.params.maxCircularity = circularity_bounds[1] if self.params.filterByCircularity else None

        self.params.filterByConvexity = convexity_bounds is not None
        self.params.minConvexity = convexity_bounds[0] if self.params.filterByConvexity else None
        self.params.maxConvexity = convexity_bounds[1] if self.params.filterByConvexity else None

        self.params.filterByInertia = inertia_bounds is not None
        self.params.minInertiaRatio = inertia_bounds[0] if self.params.filterByInertia else None
        self.params.maxInertiaRatio = inertia_bounds[1] if self.params.filterByInertia else None

        self.detector = cv2.SimpleBlobDetector.create(self.params)

    def detect(self, image: np.ndarray):
        return self.detector.detect(image)
