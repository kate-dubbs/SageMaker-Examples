
from __future__ import absolute_import

import subprocess
import sys
import io
import os

import json

import mxnet as mx
import numpy as np
import PIL.Image

# Install GluonCV so we can use the built-in FCNN transformations. 
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
    
install("gluoncv")

import gluoncv

ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()

# Pre-loading the model on startup and setting autotune to 0 for speed
net = gluoncv.model_zoo.get_model('faster_rcnn_fpn_resnet101_v1d_coco', pretrained=True, ctx=ctx)
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    """
    Load the GluonCV model. Called once when hosting service starts. Model was downloaded on startup.
    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """       
    net = gluoncv.model_zoo.get_model('faster_rcnn_fpn_resnet101_v1d_coco', pretrained=True, ctx=ctx)
    
    return net


def transform_fn(net, payload, content_type, accept):
    """
    Transform a request using the GluonCV model. Called once per request.
    :param net: The GluonCV model.
    :param payload: The request payload.
    :param content_type: The input content type.
    :param accept: The (desired) content type.
    :return: response output.
    """
    
    if content_type != 'application/x-image':
        raise RuntimeError('Content type must be application/x-image')

    f = io.BytesIO(payload)
    
    # Load image and convert to RGB space
    image = PIL.Image.open(f).convert('RGB')
    image.save('tmp.jpg')
    
    # This section is from the GluonCV tutorial for Faster-RCNN
    x, orig_img = gluoncv.data.transforms.presets.rcnn.load_test('tmp.jpg')
    x = mx.nd.array(x, ctx=ctx)
    
    box_ids, scores, bboxes = net(x)
    
    # Format the response for being passed back from the endpoint
    response = {}
    response['box_ids'] = box_ids[0].asnumpy().tolist()
    response['scores'] = scores[0].asnumpy().tolist()
    response['bboxes'] = bboxes[0].asnumpy().tolist()
    response['classes'] = net.classes
    
    output = json.dumps(response)
    
    return output

