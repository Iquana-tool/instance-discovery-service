

class ImageCacheEntry:
    def __init__(self, image):
        self.image = image  # Image in HWC format (numpy array)
        self.crop = [0., 0., 1., 1.]  # [min_x, min_y, max_x, max_y] in relative coordinates (0 to 1)

    def get_image(self):
        min_x = int(self.crop[0] * self.image.shape[1])
        min_y = int(self.crop[1] * self.image.shape[0])
        max_x = int(self.crop[2] * self.image.shape[1])
        max_y = int(self.crop[3] * self.image.shape[0])
        return self.image[min_y:max_y, min_x:max_x]

    def set_crop(self, min_x, min_y, max_x, max_y):
        assert 0 <= min_x < max_x <= 1, "Crop coordinates must be between 0 and 1 and min < max."
        assert 0 <= min_y < max_y <= 1, "Crop coordinates must be between 0 and 1 and min < max."
        self.crop = [min_x, min_y, max_x, max_y]


class ImageCache:
    def __init__(self):
        """ Simple in-memory cache for images. Maps from user ids to images. """
        self.cache = {}

    def get(self, key):
        if key not in self.cache:
            raise KeyError(f"No image found in cache for key: {key}")
        return self.cache.get(key).get_image()

    def set(self, key, image):
        self.cache[key] = ImageCacheEntry(image)

    def set_focused_crop(self, key, min_x, min_y, max_x, max_y):
        self.cache[key].set_crop(min_x, min_y, max_x, max_y)

    def set_uncropped(self, key):
        self.cache[key].set_crop(0., 0., 1., 1.)

    def clear(self):
        self.cache.clear()

    def delete(self, key):
        if key in self.cache:
            del self.cache[key]
        else:
            raise KeyError(f"No image found in cache for key: {key}")

    def __contains__(self, key):
        return key in self.cache