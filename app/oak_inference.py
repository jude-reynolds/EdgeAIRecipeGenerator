import depthai as dai
import cv2


class OakCamera:
    def __init__(self, resolution=(300, 300), fps=30):
        self.resolution = resolution
        self.fps = fps
        self.pipeline = dai.Pipeline()
        self._create_pipeline()
        self.device = None
        self.queue = None

    def _create_pipeline(self):
        cam = self.pipeline.createColorCamera()
        cam.setPreviewSize(self.resolution[0], self.resolution[1])
        cam.setInterleaved(False)
        cam.setFps(self.fps)

        xout = self.pipeline.createXLinkOut()
        xout.setStreamName('rgb')
        cam.preview.link(xout.input)

    def start(self):
        self.device = dai.Device(self.pipeline)
        self.queue = self.device.getOutputQueue(name='rgb', maxSize=4, blocking=False)
        return self

    def get(self):
        frame = self.queue.get().getCvFrame()
        return frame

    def stop(self):
        if self.device:
            self.device.close()
