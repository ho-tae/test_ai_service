from .inference_module.ai_modules import *


def main():
    # model load
    inference = Inference()

    # receiving taskID, frameID, latitude, longitude, altitude, roll, pitch, yaw, camera, img
    latitude, longitude, altitude, roll, pitch, yaw, camera = 33.204968, 126.277698, 28.5, -0.54, -3.95, 0.42, "DSC-RX100M4"
    img_path = 'example.jpg'
    img_ndarray = cv2.imread(img_path)

    # inference
    object_coords = inference(img_ndarray)
    # object_coords = [[x1,y1, x2,y2, x3,y3...class_num], [x1,y1, x2,y2, x3,y3...class_num], ...]

    # Geo-referencing
    geo_referencing = GeoReferencing([latitude, longitude, altitude, roll, pitch, yaw, camera, img_ndarray])
    inference_meta = geo_referencing(object_coords)
    # final output description:
    # inference_meta = [{ 'obj_type': 1,                                             # object class number
    #                   'obj_boundary_image' : '[x1,y1, x2,y2, x3,y3...]'            # object pixel coordinates
    #                   'obj_boundary_world': 'POLYGON ((x1,y1, x2,y2, x3,y3...))'}, # object geo-referenced coordinates
    #                   { 'obj_type': .., 'obj_boundary_image': .., 'obj_boundary_world'},...]

    logger.info(inference_meta)

    # inference result check
    inference.drawContour()


if __name__ == "__main__":
   main()




