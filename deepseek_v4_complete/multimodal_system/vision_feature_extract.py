class VisionFeatureExtractor:
    def __init__(self, backbone):
        self.backbone = backbone
    def extract(self, image):
        return self.backbone(image)
