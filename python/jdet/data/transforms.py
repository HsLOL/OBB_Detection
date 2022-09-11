import random
import jittor as jt 
import cv2
import numpy as np
import math
from PIL import Image,ImageDraw
from jdet.utils.registry import build_from_cfg,TRANSFORMS
from jdet.models.boxes.box_ops import rotated_box_to_poly_np,poly_to_rotated_box_np,norm_angle
from jdet.models.boxes.iou_calculator import bbox_overlaps_np
from numpy import random as nprandom
import time

#--- add draw label func---2022090911-----------------------
def draw_img(img,target,key = 'hboxes'):
    draw = ImageDraw.Draw(img)
    if key == 'hboxes':
        boxes = target[key]
        for box in boxes:
            x1,y1,x2,y2 = box
            draw.rectangle([x1,y1,x2,y2])
    elif key == 'polys':
        boxes = target[key]
        for box in boxes:
            x1, y1, x2, y2,x3,y3,x4,y4 = box
            draw.line([x1, y1, x2, y2])
            draw.line([x2, y2, x3, y3])
            draw.line([x3, y3, x4, y4])
            draw.line([x4, y4, x1, y1])

    elif key == 'rboxes':
        #  rrect:[x_ctr,y_ctr,w,h,angle]
        pass
    img.show()
    time.sleep(10000)
#-----------------------------------------------------

@TRANSFORMS.register_module()
class Compose:
    def __init__(self, transforms=None):
        self.transforms = []
        if transforms is None:
            transforms = []
        for transform in transforms:
            if isinstance(transform,dict):
                transform = build_from_cfg(transform,TRANSFORMS)
            elif not callable(transform):
                raise TypeError('transform must be callable or a dict')
            self.transforms.append(transform)

    def __call__(self, image, target=None):
        # ----AugmentHSV
        for t in self.transforms:
            if hasattr(t, 'name'):
                image = t(image)
            else:
                image, target = t(image, target)
        # -------------------
        #draw_img(image,target,key = 'polys')  #need to close normlize

        return image, target

@TRANSFORMS.register_module()
class RandomRotateAug:
    def __init__(self, random_rotate_on=False):
        self.random_rotate_on = random_rotate_on
    
    def _rotate_boxes_90(self,target,size):
        w,h = size
        for key in["bboxes","hboxes","rboxes","polys","hboxes_ignore","polys_ignore","rboxes_ignore"]:
            if key not in target:
                continue
            bboxes = target[key]
            if bboxes.ndim<2:
                continue
            if "bboxes" in key or "hboxes" in key:
                new_boxes = np.zeros_like(bboxes)
                new_boxes[:,  ::2] = bboxes[:, 1::2] # x = y
                # new_boxes[:, 1::2] = w - bboxes[:, -2::-2] # y = w - x
                new_boxes[:, 1] = w - bboxes[:, 2] # y = w - x
                new_boxes[:, 3] = w - bboxes[:, 0] # y = w - x
                target[key] = new_boxes
                continue

            if "rboxes" in key:
                bboxes  = rotated_box_to_poly_np(bboxes)

            new_bboxes = np.zeros_like(bboxes)
            new_bboxes[:,0::2] = bboxes[:,1::2]
            new_bboxes[:,1::2] = w-bboxes[:,0::2]

            if "rboxes" in key:
                new_bboxes = poly_to_rotated_box_np(new_bboxes)

            target[key]=new_bboxes

    def __call__( self, image, target=None ):
        # (0, 90, 180, or 270)
        if self.random_rotate_on:
            indx = int(random.random() * 100) // 25
            # anticlockwise
            for _ in range(indx):
                if target is not None:
                    self._rotate_boxes_90(target,image.size)
                image = image.rotate(90,expand=True)
            if target is not None:
                target["rotate_angle"]=90*indx

        return image, target

@TRANSFORMS.register_module()
class Resize:
    def __init__(self, min_size, max_size, keep_ratio=True):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.keep_ratio = keep_ratio

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size

        if self.keep_ratio:
            # NOTE Mingtao
            if w <= h:
              size = np.clip( size, int(w / 1.5), int(w * 1.5) )
            else:
              size = np.clip( size, int(h / 1.5), int(h * 1.5) )

            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w),1.

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)
            assert np.abs(oh/h - ow/w)<1e-2
            
        else:
            oh = self.min_size[0]
            ow = self.max_size
        
        return (oh, ow),oh/h

    def _resize_boxes(self,target,size):
        for key in ["bboxes","polys"]:
            if key not in target:
                continue
            bboxes = target[key]
            width,height = target["img_size"]
            new_w,new_h = size
            bboxes[:,0::2] = bboxes[:,0::2]*float(new_w/width)
            bboxes[:,1::2] = bboxes[:,1::2]*float(new_h/height)

            # clip to border
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, new_w - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, new_h - 1)

            target[key]=bboxes
    
    def _resize_mask(self,target,size):
        pass

    def __call__(self, image, target=None):
        size,scale_factor = self.get_size(image.size)
        image = image.resize(size[::-1],Image.BILINEAR)
        if target is not None:
            self._resize_boxes(target,image.size)
            target["img_size"] = image.size
            target["scale_factor"] = scale_factor
            target["pad_shape"] = image.size
            target["keep_ratio"] = self.keep_ratio
        return image, target

@TRANSFORMS.register_module()
class MinIoURandomCrop:
    def __init__(self,
                 min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size=0.3,
                 bbox_clip_border=True):
        # 1: return ori img
        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size
        self.bbox_clip_border = bbox_clip_border

    def __call__(self, image, target=None):
        boxes = target['bboxes']
        w, h = image.size
        while True:
            mode = nprandom.choice(self.sample_mode)
            self.mode = mode
            if mode == 1:
                return image, target

            min_iou = mode
            for i in range(50):
                new_w = nprandom.uniform(self.min_crop_size * w, w)
                new_h = nprandom.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = nprandom.uniform(w - new_w)
                top = nprandom.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                # Line or point crop is not allowed
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = bbox_overlaps_np(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue
                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                if len(overlaps) > 0:
                    # adjust boxes
                    def is_center_of_bboxes_in_patch(boxes, patch):
                        center = (boxes[:, :2] + boxes[:, 2:]) / 2
                        mask = ((center[:, 0] > patch[0]) *
                                (center[:, 1] > patch[1]) *
                                (center[:, 0] < patch[2]) *
                                (center[:, 1] < patch[3]))
                        return mask

                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if not mask.any():
                        continue
                    boxes = boxes[mask]
                    if self.bbox_clip_border:
                        boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                        boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                    boxes -= np.tile(patch[:2], 2)
                    target['bboxes'] = boxes
                    # labels
                    target['labels'] = target['labels'][mask]

                # adjust the img no matter whether the gt is empty before crop
                image_crop = image.crop(patch)
                target['img_size'] = image_crop.size

                return image_crop, target


@TRANSFORMS.register_module()
class Expand:
    def __init__(self,
                 mean=(0, 0, 0),
                 ratio_range=(1, 4),
                 prob=0.5):
        self.ratio_range = ratio_range
        self.mean = tuple([int(i) for i in mean])
        self.min_ratio, self.max_ratio = ratio_range
        self.prob = prob

    def __call__(self, image, target=None):
        if nprandom.uniform(0, 1) > self.prob:
            return image, target
        w, h = image.size
        ratio = nprandom.uniform(self.min_ratio, self.max_ratio)
        left = int(nprandom.uniform(0, w * ratio - w))
        top = int(nprandom.uniform(0, h * ratio - h))
        
        new_image = Image.new(image.mode,(int(w * ratio), int(h * ratio)), self.mean)
        new_image.paste(image,(left, top, left+image.size[0], top+image.size[1]))

        target["bboxes"] = target["bboxes"] + np.tile(
            (left, top), 2).astype(target["bboxes"].dtype)
        target['img_size'] = new_image.size
        return new_image, target

@TRANSFORMS.register_module()
class PhotoMetricDistortion:
    def __init__(self, brightness_delta=32./255, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18):
        self.t = jt.transform.ColorJitter(brightness=brightness_delta, contrast=contrast_range, saturation=saturation_range, hue=hue_delta)

    def __call__(self, image, target=None):
        image = self.t(image)
        
        return image, target

@TRANSFORMS.register_module()
class Resize_keep_ratio:
    def __init__(self, min_size, max_size, keep_ratio=True):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.keep_ratio = keep_ratio

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size

        oh = self.min_size[0]
        ow = self.max_size
        return (oh, ow), [ow/w, oh/h, ow/w, oh/h]
        

    def _resize_boxes(self,target,size):
        for key in ["bboxes","polys"]:
            if key not in target:
                continue
            bboxes = target[key]
            width,height = target["img_size"]
            new_w,new_h = size
            bboxes[:,0::2] = bboxes[:,0::2]*float(new_w/width)
            bboxes[:,1::2] = bboxes[:,1::2]*float(new_h/height)

            # clip to border
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, new_w - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, new_h - 1)

            target[key]=bboxes
    
    def _resize_mask(self,target,size):
        pass

    def __call__(self, image, target=None):
        size,scale_factor = self.get_size(image.size)
        image = image.resize(size[::-1],Image.BILINEAR)
        if target is not None:
            self._resize_boxes(target,image.size)
            target["img_size"] = image.size
            target["scale_factor"] = scale_factor
            target["pad_shape"] = image.size
            target["keep_ratio"] = self.keep_ratio
        return image, target


@TRANSFORMS.register_module()
class RotatedResize(Resize):

    def _resize_boxes(self, target,size):
        for key in ["bboxes","hboxes","rboxes","polys","hboxes_ignore","polys_ignore","rboxes_ignore"]:
            if key not in target:
                continue
            bboxes = target[key]
            if bboxes is None or bboxes.ndim!=2:
                continue
            
            if "rboxes" in key:
                bboxes = rotated_box_to_poly_np(bboxes)

            width,height = target["img_size"]
            new_w,new_h = size
            
            bboxes[:,0::2] = bboxes[:,0::2]*float(new_w/width)
            bboxes[:,1::2] = bboxes[:,1::2]*float(new_h/height)

            # clip to border
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, new_w - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, new_h - 1)
            
            if "rboxes" in key:
                bboxes = poly_to_rotated_box_np(bboxes)
            target[key]=bboxes


@TRANSFORMS.register_module()
class RandomFlip:
    def __init__(self, prob=0.5,direction="horizontal"):
        assert direction in ['horizontal', 'vertical', 'diagonal'],f"{direction} not supported"
        self.direction = direction
        self.prob = prob
    
    def _flip_boxes(self,target,size):
        w,h = target["img_size"] 
        for key in ["bboxes","polys"]:
            if key not in target:
                continue
            bboxes = target[key]
            flipped = bboxes.copy()
            if self.direction == 'horizontal':
                flipped[..., 0::4] = w - bboxes[..., 2::4]
                flipped[..., 2::4] = w - bboxes[..., 0::4]
            elif self.direction == 'vertical':
                flipped[..., 1::4] = h - bboxes[..., 3::4]
                flipped[..., 3::4] = h - bboxes[..., 1::4]
            elif self.direction == 'diagonal':
                flipped[..., 0::4] = w - bboxes[..., 2::4]
                flipped[..., 1::4] = h - bboxes[..., 3::4]
                flipped[..., 2::4] = w - bboxes[..., 0::4]
                flipped[..., 3::4] = h - bboxes[..., 1::4]
            target[key] = flipped

    def _flip_image(self,image):
        if self.direction=="horizontal":
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif self.direction == "vertical":
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif self.direction == "diagonal":
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = self._flip_image(image)
            if target is not None:
                self._flip_boxes(target,image.size)
            target["flip"]=self.direction
        return image, target

@TRANSFORMS.register_module()
class RotatedRandomFlip(RandomFlip):
    def _flip_rboxes(self,bboxes,w,h):
        flipped = bboxes.copy()
        if self.direction == 'horizontal':
            flipped[..., 0::5] = w - flipped[..., 0::5] - 1
            flipped[..., 4::5] = norm_angle(np.pi - flipped[..., 4::5])
        elif self.direction == 'vertical':
            flipped[..., 1::5] = h - flipped[..., 1::5] - 1
            flipped[..., 4::5] = norm_angle( - flipped[..., 4::5])
        elif self.direction == 'diagonal':
            assert False
        else:
            assert False
        return flipped

    def _flip_polys(self,bboxes,w,h):
        flipped = bboxes.copy()
        if self.direction == 'horizontal':
            flipped[..., 0::2] = w - flipped[..., 0::2] - 1
        elif self.direction == 'vertical':
            flipped[..., 1::2] = h - flipped[..., 1::2] - 1
        elif self.direction == 'diagonal':
            flipped[..., 0::2] = w - flipped[..., 0::2] - 1
            flipped[..., 1::2] = h - flipped[..., 1::2] - 1
        return flipped 


    def _flip_boxes(self,target,size):
        w,h = size 
        for key in ["bboxes","hboxes","rboxes","polys","hboxes_ignore","polys_ignore","rboxes_ignore"]:
            if key not in target:
                continue
            bboxes = target[key]
            if "rboxes" in key:
                target[key] = self._flip_rboxes(bboxes,w,h)
                continue 
            if "polys" in key:
                target[key] = self._flip_polys(bboxes,w,h)
                continue
            flipped = bboxes.copy()
            if self.direction == 'horizontal':
                flipped[..., 0::4] = w - bboxes[..., 2::4]
                flipped[..., 2::4] = w - bboxes[..., 0::4]
            elif self.direction == 'vertical':
                flipped[..., 1::4] = h - bboxes[..., 3::4]
                flipped[..., 3::4] = h - bboxes[..., 1::4]
            elif self.direction == 'diagonal':
                flipped[..., 0::4] = w - bboxes[..., 2::4]
                flipped[..., 1::4] = h - bboxes[..., 3::4]
                flipped[..., 2::4] = w - bboxes[..., 0::4]
                flipped[..., 3::4] = h - bboxes[..., 1::4]
            target[key] = flipped

@TRANSFORMS.register_module()
class Pad:
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def __call__(self,image,target=None):
        if self.size is not None:
            pad_w,pad_h = self.size
        else:
            pad_h = int(np.ceil(image.size[1] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(image.size[0] / self.size_divisor)) * self.size_divisor
        
        new_image = Image.new(image.mode,(pad_w,pad_h),(self.pad_val,)*len(image.split()))
        new_image.paste(image,(0,0,image.size[0],image.size[1]))
        target["pad_shape"] = new_image.size
        
        return new_image,target
    

@TRANSFORMS.register_module()
class Normalize:
    def __init__(self, mean, std, to_bgr=True):
        self.mean = np.float32(mean).reshape(-1,1,1)
        self.std = np.float32(std).reshape(-1,1,1)
        self.to_bgr = to_bgr

    def __call__(self, image, target=None):
        if isinstance(image,Image.Image):
            image = np.array(image).transpose((2,0,1))

        if self.to_bgr:
            image = image[::-1]
        
        image = (image - self.mean) / self.std

        target["mean"] = self.mean 
        target["std"] = self.std
        target["to_bgr"] = self.to_bgr
        return image, target


@TRANSFORMS.register_module()
class YoloRandomPerspective:
    def __init__ (self, degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0):
        self.degrees=degrees
        self.translate=translate
        self.scale=scale
        self.shear=shear
        self.perspective=perspective
    
    def __call__(self, img, targets=(), border=(0, 0)):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # targets = [cls, xyxy]

        height = img.shape[0] + border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        # Visualize
        # import matplotlib.pyplot as plt
        # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
        # ax[0].imshow(img[:, :, ::-1])  # base
        # ax[1].imshow(img2[:, :, ::-1])  # warped

        # Transform label coordinates
        n = len(targets)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            if self.perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            else:  # affine
                xy = xy[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # # apply angle-based reduction of bounding boxes
            # radians = a * math.pi / 180
            # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            # x = (xy[:, 2] + xy[:, 0]) / 2
            # y = (xy[:, 3] + xy[:, 1]) / 2
            # w = (xy[:, 2] - xy[:, 0]) * reduction
            # h = (xy[:, 3] - xy[:, 1]) * reduction
            # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

            # filter candidates
            i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
            targets = targets[i]
            targets[:, 1:5] = xy[i]

        return img, targets

@TRANSFORMS.register_module()
class YoloAugmentHSV:
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        self.hgain=hgain 
        self.sgain=sgain
        self.vgain=vgain 

    def __call__(self, img):
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

@TRANSFORMS.register_module()
class YoloRandomFlip:
    def __init__(self, prob=0.5, direction="horizontal"):
        assert direction in ['horizontal', 'vertical', 'diagonal'],f"{direction} not supported"
        self.direction = direction
        self.prob = prob

    def _flip_image(self, image):
        if self.direction=="horizontal":
            image = np.fliplr(image)
        elif self.direction=="vertical":
            image = np.flipud(image)
        elif self.direction == "diagonal":
            image = np.fliplr(image)
            image = np.flipud(image)
        return image

    def _flip_boxes(self, target):
        if self.direction=='horizontal':
            target[:, 1] = 1 - target[:, 1]
        elif self.direction=='vertical':
            labels[:, 2] = 1 - labels[:, 2]
        elif self.direction=='diagonal':
            target[:, 1] = 1 - target[:, 1]
            labels[:, 2] = 1 - labels[:, 2]
        
    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = self._flip_image(image)
            if target is not None:
                self._flip_boxes(target)
        return image, target

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


#------add datacheck ---2022090911-----------------------
@TRANSFORMS.register_module()
class DataCheck:
    """Filter out small bboxes and outside bboxes"""
    def __init__(self,min_bbox_w = 10,min_bbox_h = 10,min_bbox_area = 80):
        self.min_bbox_w = min_bbox_w
        self.min_bbox_h = min_bbox_h
        self.min_bbox_area = min_bbox_area


    def __call__(self, image, target=None):

        #-----get valid box idx
        bboxes = target["rboxes"]
        bbox_w, bbox_h = bboxes[..., 2], bboxes[..., 3]
        valid_inds =  (bbox_w > self.min_bbox_w) & \
                      (bbox_h > self.min_bbox_h) & \
                      (bbox_h * bbox_w >= self.min_bbox_area)
        valid_inds = np.nonzero(valid_inds)[0]
        for key in ["bboxes","hboxes","rboxes","polys"]:
            if key not in target:
                continue
            target[key] = target[key][valid_inds]
        return image,target
#------------------------------------------------------

#------add HSV Augment ---20220909
@TRANSFORMS.register_module()
class AugmentHSV:
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        self.hgain=hgain
        self.sgain=sgain
        self.vgain=vgain
        self.name = 'HSVAug'

    def __call__(self, img):
        #pil to cv2
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

        #cv2 to pil
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        return img
#----------------------

# #------add mosaic ---2022090910-----------------------
# import copy
#
# def find_inside_bboxes(bboxes, img_h, img_w):
#     """Find bboxes as long as a part of bboxes is inside the image.
#
#     Args:
#         bboxes (Tensor): Shape (N, 4).
#         img_h (int): Image height.
#         img_w (int): Image width.
#
#     Returns:
#         Tensor: Index of the remaining bboxes.
#     """
#     inside_inds = (bboxes[:, 0] < img_w) & (bboxes[:, 2] > 0) \
#         & (bboxes[:, 1] < img_h) & (bboxes[:, 3] > 0)
#     return inside_inds
#


# @TRANSFORMS.register_module()
# class Mosaic:
#     """Mosaic augmentation.
#
#     Given 4 images, mosaic transform combines them into
#     one output image. The output image is composed of the parts from each sub-
#     image.
#
#     .. code:: text
#
#                         mosaic transform
#                            center_x
#                 +------------------------------+
#                 |       pad        |  pad      |
#                 |      +-----------+           |
#                 |      |           |           |
#                 |      |  image1   |--------+  |
#                 |      |           |        |  |
#                 |      |           | image2 |  |
#      center_y   |----+-------------+-----------|
#                 |    |   cropped   |           |
#                 |pad |   image3    |  image4   |
#                 |    |             |           |
#                 +----|-------------+-----------+
#                      |             |
#                      +-------------+
#
#      The mosaic transform steps are as follows:
#
#          1. Choose the mosaic center as the intersections of 4 images
#          2. Get the left top image according to the index, and randomly
#             sample another 3 images from the custom dataset.
#          3. Sub image will be cropped if image is larger than mosaic patch
#
#     Args:
#         img_scale (Sequence[int]): Image size after mosaic pipeline of single
#             image. The shape order should be (height, width).
#             Default to (640, 640).
#         center_ratio_range (Sequence[float]): Center ratio range of mosaic
#             output. Default to (0.5, 1.5).
#         min_bbox_size (int | float): The minimum pixel for filtering
#             invalid bboxes after the mosaic pipeline. Default to 0.
#         bbox_clip_border (bool, optional): Whether to clip the objects outside
#             the border of the image. In some dataset like MOT17, the gt bboxes
#             are allowed to cross the border of images. Therefore, we don't
#             need to clip the gt bboxes in these cases. Defaults to True.
#         skip_filter (bool): Whether to skip filtering rules. If it
#             is True, the filter rule will not be applied, and the
#             `min_bbox_size` is invalid. Default to True.
#         pad_val (int): Pad value. Default to 114.
#         prob (float): Probability of applying this transformation.
#             Default to 1.0.
#     """
#
#     def __init__(self,
#                  img_scale=(640, 640),
#                  center_ratio_range=(0.5, 1.5),
#                  min_bbox_size=0,
#                  bbox_clip_border=True,
#                  skip_filter=True,
#                  pad_val=114,
#                  prob=1.0):
#         assert isinstance(img_scale, tuple)
#         assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. '\
#             f'got {prob}.'
#
#         #log_img_scale(img_scale, skip_square=True)
#         self.img_scale = img_scale
#         self.center_ratio_range = center_ratio_range
#         self.min_bbox_size = min_bbox_size
#         self.bbox_clip_border = bbox_clip_border
#         self.skip_filter = skip_filter
#         self.pad_val = pad_val
#         self.prob = prob
#
#     def __call__(self, results):
#         """Call function to make a mosaic of image.
#
#         Args:
#             results (dict): Result dict.
#
#         Returns:
#             dict: Result dict with mosaic transformed.
#         """
#
#         if random.uniform(0, 1) > self.prob:
#             return results
#
#         results = self._mosaic_transform(results)
#         return results
#
#     def get_indexes(self, dataset):
#         """Call function to collect indexes.
#
#         Args:
#             dataset (:obj:`MultiImageMixDataset`): The dataset.
#
#         Returns:
#             list: indexes.
#         """
#
#         indexes = [random.randint(0, len(dataset)) for _ in range(3)]
#         return indexes
#
#     def _mosaic_transform(self, results):
#         """Mosaic transform function.
#
#         Args:
#             results (dict): Result dict.
#
#         Returns:
#             dict: Updated result dict.
#         """
#
#         assert 'mix_results' in results
#         mosaic_labels = []
#         mosaic_bboxes = []
#         if len(results['img'].shape) == 3:
#             mosaic_img = np.full(
#                 (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
#                 self.pad_val,
#                 dtype=results['img'].dtype)
#         else:
#             mosaic_img = np.full(
#                 (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
#                 self.pad_val,
#                 dtype=results['img'].dtype)
#
#         # mosaic center x, y
#         center_x = int(
#             random.uniform(*self.center_ratio_range) * self.img_scale[1])
#         center_y = int(
#             random.uniform(*self.center_ratio_range) * self.img_scale[0])
#         center_position = (center_x, center_y)
#
#         loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
#         for i, loc in enumerate(loc_strs):
#             if loc == 'top_left':
#                 results_patch = copy.deepcopy(results)
#             else:
#                 results_patch = copy.deepcopy(results['mix_results'][i - 1])
#
#             img_i = results_patch['img']
#             h_i, w_i = img_i.shape[:2]
#             # keep_ratio resize
#             scale_ratio_i = min(self.img_scale[0] / h_i,
#                                 self.img_scale[1] / w_i)
#             img_i = cv2.imresize(
#                 img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))
#
#             # compute the combine parameters
#             paste_coord, crop_coord = self._mosaic_combine(
#                 loc, center_position, img_i.shape[:2][::-1])
#             x1_p, y1_p, x2_p, y2_p = paste_coord
#             x1_c, y1_c, x2_c, y2_c = crop_coord
#
#             # crop and paste image
#             mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]
#
#             # adjust coordinate
#             gt_bboxes_i = results_patch['gt_bboxes']
#             gt_labels_i = results_patch['gt_labels']
#
#             if gt_bboxes_i.shape[0] > 0:
#                 padw = x1_p - x1_c
#                 padh = y1_p - y1_c
#                 gt_bboxes_i[:, 0::2] = \
#                     scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
#                 gt_bboxes_i[:, 1::2] = \
#                     scale_ratio_i * gt_bboxes_i[:, 1::2] + padh
#
#             mosaic_bboxes.append(gt_bboxes_i)
#             mosaic_labels.append(gt_labels_i)
#
#         if len(mosaic_labels) > 0:
#             mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
#             mosaic_labels = np.concatenate(mosaic_labels, 0)
#
#             if self.bbox_clip_border:
#                 mosaic_bboxes[:, 0::2] = np.clip(mosaic_bboxes[:, 0::2], 0,
#                                                  2 * self.img_scale[1])
#                 mosaic_bboxes[:, 1::2] = np.clip(mosaic_bboxes[:, 1::2], 0,
#                                                  2 * self.img_scale[0])
#
#             if not self.skip_filter:
#                 mosaic_bboxes, mosaic_labels = \
#                     self._filter_box_candidates(mosaic_bboxes, mosaic_labels)
#
#         # remove outside bboxes
#         inside_inds = find_inside_bboxes(mosaic_bboxes, 2 * self.img_scale[0],
#                                          2 * self.img_scale[1])
#         mosaic_bboxes = mosaic_bboxes[inside_inds]
#         mosaic_labels = mosaic_labels[inside_inds]
#
#         results['img'] = mosaic_img
#         results['img_shape'] = mosaic_img.shape
#         results['gt_bboxes'] = mosaic_bboxes
#         results['gt_labels'] = mosaic_labels
#
#         return results
#
#     def _mosaic_combine(self, loc, center_position_xy, img_shape_wh):
#         """Calculate global coordinate of mosaic image and local coordinate of
#         cropped sub-image.
#
#         Args:
#             loc (str): Index for the sub-image, loc in ('top_left',
#               'top_right', 'bottom_left', 'bottom_right').
#             center_position_xy (Sequence[float]): Mixing center for 4 images,
#                 (x, y).
#             img_shape_wh (Sequence[int]): Width and height of sub-image
#
#         Returns:
#             tuple[tuple[float]]: Corresponding coordinate of pasting and
#                 cropping
#                 - paste_coord (tuple): paste corner coordinate in mosaic image.
#                 - crop_coord (tuple): crop corner coordinate in mosaic image.
#         """
#         assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
#         if loc == 'top_left':
#             # index0 to top left part of image
#             x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
#                              max(center_position_xy[1] - img_shape_wh[1], 0), \
#                              center_position_xy[0], \
#                              center_position_xy[1]
#             crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
#                 y2 - y1), img_shape_wh[0], img_shape_wh[1]
#
#         elif loc == 'top_right':
#             # index1 to top right part of image
#             x1, y1, x2, y2 = center_position_xy[0], \
#                              max(center_position_xy[1] - img_shape_wh[1], 0), \
#                              min(center_position_xy[0] + img_shape_wh[0],
#                                  self.img_scale[1] * 2), \
#                              center_position_xy[1]
#             crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
#                 img_shape_wh[0], x2 - x1), img_shape_wh[1]
#
#         elif loc == 'bottom_left':
#             # index2 to bottom left part of image
#             x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
#                              center_position_xy[1], \
#                              center_position_xy[0], \
#                              min(self.img_scale[0] * 2, center_position_xy[1] +
#                                  img_shape_wh[1])
#             crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
#                 y2 - y1, img_shape_wh[1])
#
#         else:
#             # index3 to bottom right part of image
#             x1, y1, x2, y2 = center_position_xy[0], \
#                              center_position_xy[1], \
#                              min(center_position_xy[0] + img_shape_wh[0],
#                                  self.img_scale[1] * 2), \
#                              min(self.img_scale[0] * 2, center_position_xy[1] +
#                                  img_shape_wh[1])
#             crop_coord = 0, 0, min(img_shape_wh[0],
#                                    x2 - x1), min(y2 - y1, img_shape_wh[1])
#
#         paste_coord = x1, y1, x2, y2
#         return paste_coord, crop_coord
#
#     def _filter_box_candidates(self, bboxes, labels):
#         """Filter out bboxes too small after Mosaic."""
#         bbox_w = bboxes[:, 2] - bboxes[:, 0]
#         bbox_h = bboxes[:, 3] - bboxes[:, 1]
#         valid_inds = (bbox_w > self.min_bbox_size) & \
#                      (bbox_h > self.min_bbox_size)
#         valid_inds = np.nonzero(valid_inds)[0]
#         return bboxes[valid_inds], labels[valid_inds]
#
#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'img_scale={self.img_scale}, '
#         repr_str += f'center_ratio_range={self.center_ratio_range}, '
#         repr_str += f'pad_val={self.pad_val}, '
#         repr_str += f'min_bbox_size={self.min_bbox_size}, '
#         repr_str += f'skip_filter={self.skip_filter})'
#         return repr_str
#
#
#
# @TRANSFORMS.register_module()
# class RMosaic(Mosaic):
#     """Rotate Mosaic augmentation. Inherit from
#     `mmdet.datasets.pipelines.transforms.Mosaic`.
#
#     Given 4 images, mosaic transform combines them into
#     one output image. The output image is composed of the parts from each sub-
#     image.
#
#     .. code:: text
#                         mosaic transform
#                            center_x
#                 +------------------------------+
#                 |       pad        |  pad      |
#                 |      +-----------+           |
#                 |      |           |           |
#                 |      |  image1   |--------+  |
#                 |      |           |        |  |
#                 |      |           | image2 |  |
#      center_y   |----+-------------+-----------|
#                 |    |   cropped   |           |
#                 |pad |   image3    |  image4   |
#                 |    |             |           |
#                 +----|-------------+-----------+
#                      |             |
#                      +-------------+
#
#      The mosaic transform steps are as follows:
#          1. Choose the mosaic center as the intersections of 4 images
#          2. Get the left top image according to the index, and randomly
#             sample another 3 images from the custom dataset.
#          3. Sub image will be cropped if image is larger than mosaic patch
#
#     Args:
#         img_scale (Sequence[int]): Image size after mosaic pipeline of single
#             image. The shape order should be (height, width).
#             Defaults to (640, 640).
#         center_ratio_range (Sequence[float]): Center ratio range of mosaic
#             output. Defaults to (0.5, 1.5).
#         min_bbox_size (int | float): The minimum pixel for filtering
#             invalid bboxes after the mosaic pipeline. Defaults to 0.
#         bbox_clip_border (bool, optional): Whether to clip the objects outside
#             the border of the image. In some dataset like MOT17, the gt bboxes
#             are allowed to cross the border of images. Therefore, we don't
#             need to clip the gt bboxes in these cases. Defaults to True.
#         skip_filter (bool): Whether to skip filtering rules. If it
#             is True, the filter rule will not be applied, and the
#             `min_bbox_size` is invalid. Defaults to True.
#         pad_val (int): Pad value. Defaults to 114.
#         prob (float): Probability of applying this transformation.
#             Defaults to 1.0.
#         version  (str, optional): Angle representations. Defaults to `oc`.
#     """
#
#     def __init__(self,
#                  img_scale=(640, 640),
#                  center_ratio_range=(0.5, 1.5),
#                  min_bbox_size=10,
#                  bbox_clip_border=True,
#                  skip_filter=True,
#                  pad_val=114,
#                  prob=1.0,
#                  version='oc'):
#         super(RMosaic, self).__init__(
#             img_scale=img_scale,
#             center_ratio_range=center_ratio_range,
#             min_bbox_size=min_bbox_size,
#             bbox_clip_border=bbox_clip_border,
#             skip_filter=skip_filter,
#             pad_val=pad_val,
#             prob=1.0)
#
#     def _mosaic_transform(self, img,target):
#         """Mosaic transform function.
#
#         Args:
#             results (dict): Result dict.
#
#         Returns:
#             dict: Updated result dict.
#         """
#         results = {}
#         results['img'] = img
#         results['img'] = img
#
#
#
#         mosaic_labels = []
#         mosaic_bboxes = []
#         if len(results['img'].shape) == 3:
#             mosaic_img = np.full(
#                 (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
#                 self.pad_val,
#                 dtype=results['img'].dtype)
#         else:
#             mosaic_img = np.full(
#                 (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
#                 self.pad_val,
#                 dtype=results['img'].dtype)
#
#         # mosaic center x, y
#         center_x = int(
#             random.uniform(*self.center_ratio_range) * self.img_scale[1])
#         center_y = int(
#             random.uniform(*self.center_ratio_range) * self.img_scale[0])
#         center_position = (center_x, center_y)
#
#         loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
#         for i, loc in enumerate(loc_strs):
#             if loc == 'top_left':
#                 results_patch = copy.deepcopy(results)
#             else:
#                 results_patch = copy.deepcopy(results['mix_results'][i - 1])
#
#             img_i = results_patch['img']
#             h_i, w_i = img_i.shape[:2]
#             # keep_ratio resize
#             scale_ratio_i = min(self.img_scale[0] / h_i,
#                                 self.img_scale[1] / w_i)
#             img_i = cv2.imresize(
#                 img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))
#
#             # compute the combine parameters
#             paste_coord, crop_coord = self._mosaic_combine(
#                 loc, center_position, img_i.shape[:2][::-1])
#             x1_p, y1_p, x2_p, y2_p = paste_coord
#             x1_c, y1_c, x2_c, y2_c = crop_coord
#
#             # crop and paste image
#             mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]
#
#             # adjust coordinate
#             gt_bboxes_i = results_patch['gt_bboxes']
#             gt_labels_i = results_patch['gt_labels']
#
#             if gt_bboxes_i.shape[0] > 0:
#                 padw = x1_p - x1_c
#                 padh = y1_p - y1_c
#                 gt_bboxes_i[:, 0] = \
#                     scale_ratio_i * gt_bboxes_i[:, 0] + padw
#                 gt_bboxes_i[:, 1] = \
#                     scale_ratio_i * gt_bboxes_i[:, 1] + padh
#                 gt_bboxes_i[:, 2:4] = \
#                     scale_ratio_i * gt_bboxes_i[:, 2:4]
#
#             mosaic_bboxes.append(gt_bboxes_i)
#             mosaic_labels.append(gt_labels_i)
#
#         if len(mosaic_labels) > 0:
#             mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
#             mosaic_labels = np.concatenate(mosaic_labels, 0)
#
#             mosaic_bboxes, mosaic_labels = \
#                 self._filter_box_candidates(
#                     mosaic_bboxes, mosaic_labels,
#                     2 * self.img_scale[1], 2 * self.img_scale[0]
#                 )
#         # If results after rmosaic does not contain any valid gt-bbox,
#         # return None. And transform flows in MultiImageMixDataset will
#         # repeat until existing valid gt-bbox.
#         if len(mosaic_bboxes) == 0:
#             return None
#
#         results['img'] = mosaic_img
#         cv2.imshow('a',mosaic_img)
#         cv2.Waitekey()
#         results['img_shape'] = mosaic_img.shape
#         results['gt_bboxes'] = mosaic_bboxes
#         results['gt_labels'] = mosaic_labels
#
#         return results
#
#     def _filter_box_candidates(self, bboxes, labels, w, h):
#         """Filter out small bboxes and outside bboxes after Mosaic."""
#         bbox_x, bbox_y, bbox_w, bbox_h = \
#             bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
#         valid_inds = (bbox_x > 0) & (bbox_x < w) & \
#                      (bbox_y > 0) & (bbox_y < h) & \
#                      (bbox_w > self.min_bbox_size) & \
#                      (bbox_h > self.min_bbox_size)
#         valid_inds = np.nonzero(valid_inds)[0]
#         return bboxes[valid_inds], labels[valid_inds]

