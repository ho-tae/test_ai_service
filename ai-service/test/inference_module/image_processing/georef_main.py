from image_processing import drones
import image_processing.georef_for_eo as georeferencers
from image_processing.georef_for_gp import Rot3D,georef_inference,create_inference_metadata, geographic2plane
import numpy as np

# taskID, frameIndex, latitude, longitude, altitude, roll, pitch, yaw, camera, img = receive(sock_s)

class GeoReferencing:
    def __init__(self, metadata):
        self.latitude, self.longitude, self.altitude, self.roll, self.pitch, self.yaw, self.camera, self.img = metadata

        # self.my_drone = drones.Drones(make=self.camera, pre_calibrated=False)
        self.my_drone = drones.Drones(make=self.camera)
        self.pre_calibrated = False
        self.ground_height = 0.0
        self.adjusted_eo = 0

    def system_calibration(self):
        init_eo = np.array([self.longitude, self.latitude, self.altitude, self.roll, self.pitch, self.yaw])
        init_eo[:2] = geographic2plane(init_eo, 3857)
        # if self.my_drone.pre_calibrated:
        if self.pre_calibrated:
            init_eo[3:] *= np.pi / 180
            adjusted_eo = init_eo
        else:
            my_georeferencer = georeferencers.DirectGeoreferencer()
            adjusted_eo = my_georeferencer.georeference(self.my_drone, init_eo)
        return adjusted_eo

    def __call__(self,object_coords):
        adjusted_eo = self.system_calibration()
        img_rows = self.img.shape[0]
        img_cols = self.img.shape[1]
        pixel_size = self.my_drone.sensor_width / img_cols  # mm/px
        R_CG = Rot3D(adjusted_eo).T

        inference_metadata = []
        for inference_px in object_coords:
            inference_world = georef_inference(inference_px[:-1], img_rows, img_cols, pixel_size,
                                               self.my_drone.focal_length, adjusted_eo, R_CG, self.ground_height)
            inference_metadata.append(create_inference_metadata(inference_px[-1], inference_px[:-1], inference_world))

        return inference_metadata











