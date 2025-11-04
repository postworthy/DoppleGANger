# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 


from __future__ import division

import gc
import glob
import os.path as osp

import numpy as np
import onnxruntime
from numpy.linalg import norm

from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX
from face import Face

__all__ = ['FaceAnalysis']

class FaceAnalysis:
    _instance = None

    def __new__(cls, *args, **kwargs):
        # If we already have an instance, return it
        if cls._instance is not None:
            return cls._instance
        # Otherwise, create a new instance and store it
        cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def is_initialized(cls):
        # True if there's an instance and it has _initialized == True
        return (
            cls._instance is not None and 
            getattr(cls._instance, '_initialized', False)
        )
    
    @classmethod
    def destroy(cls):
        """
        Destroy the singleton instance and clear references, so memory can be freed.
        This includes GPU memory if onnxruntime was using CUDA. 
        The next time you call INSwapper(...), it will create a fresh instance.
        """
        if cls._instance is not None:
            # Drop references to large resources
            if hasattr(cls._instance, 'det_model'):
                cls._instance.det_model.destroy()
                del cls._instance.det_model

            if hasattr(cls._instance, 'arcface_model'):
                cls._instance.arcface_model.destroy()
                del cls._instance.arcface_model
            # If you have other large attributes, clear them out here as well, e.g.:
            # del cls._instance.emap

        cls._instance = None
        # Force a garbage collection to attempt freeing GPU memory sooner
        gc.collect()
    
    def __init__(self, root='/root/.insightface', allowed_modules=None, **kwargs):
        # If already initialized (i.e. fields are set), do nothing
        # so we only run init logic exactly once
        if getattr(self, '_initialized', False):
            return
        self._initialized = True

        onnxruntime.set_default_logger_severity(3)
        self.det_model = SCRFD(root+"/models/buffalo_l/det_10g.onnx")
        self.arcface_model = ArcFaceONNX(root+"/models/buffalo_l/w600k_r50.onnx")

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        self.det_model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
        self.arcface_model.prepare(ctx_id)

    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            self.arcface_model.get(img, face)
            ret.append(face)
        return ret

    def draw_on(self, img, faces):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(int)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(int)
                #print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                               2)
            if face.gender is not None and face.age is not None:
                cv2.putText(dimg,'%s,%d'%(face.sex,face.age), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

        return dimg
