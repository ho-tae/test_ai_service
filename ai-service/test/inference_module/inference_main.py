from mmdet.apis import init_detector, inference_detector
from mmdet.datasets.imgprocess import server_det_bboxes, server_det_masks
import mmcv, cv2
from skimage import draw
import numpy as np

class Inference:
    def __init__(self):
        config = "inference_module/configs/cascade_rcnn_x101_64x4d_fpn_1x_coco_cbp.py"
        model_ckpt = "inference_module/configs/epoch_20_cbp.pth"
        self.score_thr = 0.1
        self.CLASSES = ('people','people')
        self.model = self.inference_module_init(config, model_ckpt)

    def inference_module_init(self, config_file, checkpoint_file):
        model = init_detector(config_file, checkpoint_file, device='cuda:0')
        return model

    def drawContour(self):
        for bbox in self.object_coords:
            bbox_coord = bbox[:-1]
            n = 2
            contour = np.array([[bbox_coord[i * n:(i + 1) * n]] for i in range((len(bbox_coord) + n - 1) // n )]).astype('int32')
            cv2.drawContours(self.image,[contour],0,(0,0,0),2)

        overlay_img = cv2.resize(self.image, (1920, 1080))
        # cv2.imwrite('configs/example_result.jpg',overlay_img)
        cv2.imshow('image', overlay_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __call__(self, img):
        self.image = img
        result = inference_detector(self.model, self.image)
        self.object_coords = server_det_bboxes(img, result, score_thr=0.3)
        return self.object_coords
