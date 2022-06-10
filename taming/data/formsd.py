import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
from taming.data.base import ImagePaths, NumpyPaths
from PIL import Image
from queue import Queue
from taming.data.faceshq import FacesBase

def rescaleCrop(imgfl,tileSize=128):
    img = Image.open(imgfl).convert("RGB")
    w, h = img.size
    img = img.resize(((w // 2) * 2, (h // 2) * 2), resample=Image.LANCZOS)
    im = np.array(img).astype(np.uint8)
    vtt = 18
    htt = 14
    hxw_tiles = (np.array(im.shape[:-1]) + (tileSize - 1)) // tileSize
    hxw_tiles = np.max([hxw_tiles, np.array([(vtt - 1), (htt - 1)])], axis=0)
    new_height = hxw_tiles[0] * tileSize
    new_width = hxw_tiles[1] * tileSize
    if hxw_tiles[0] >= vtt or hxw_tiles[1] >= htt:  # crop to 17x13 tiles
        new_height = (vtt - 1) * tileSize
        new_width = (htt - 1) * tileSize
    canvas = np.zeros([new_height, new_width, 3], dtype=im.dtype)
    dst_yx_o = (np.array([new_height, new_width]) - np.array(im.shape[:-1])) // 2
    copy_height = min(new_height, im.shape[0])
    copy_width = min(new_width, im.shape[1])
    # print(im.dtype, im.strides, im.shape, canvas.shape, dst_yx_o, list(hxw_tiles))
    src_yx_o = np.clip(-dst_yx_o, 0, np.array(im.shape[:-1]).max())
    dst_yx_o = np.clip(dst_yx_o, 0, np.array(im.shape[:-1]).max())
    # print(copy_height,copy_width,im.shape,src_yx_o,dst_yx_o)
    canvas[dst_yx_o[0]:dst_yx_o[0] + copy_height,
    dst_yx_o[1]:dst_yx_o[1] + copy_width, :] = im[src_yx_o[0]:im.shape[0] - src_yx_o[0],
                                               src_yx_o[1]:im.shape[1] - src_yx_o[1], :]
    canvas = (canvas / 127.5 - 1.0).astype(np.float32)
    return canvas

def splitImage(im, tileSize=128):
    hxw_tiles = (np.array(im.shape[:-1]) + (tileSize - 1)) // tileSize
    # print(im.dtype,im.strides,im.shape,list(hxw_tiles))
    _strides = im.strides

    # plt.imshow(canvas)
    # plt.show()
    # cv2.imshow(canvas)
    tiles = np.lib.stride_tricks.as_strided(im, shape=list(hxw_tiles) + [tileSize, tileSize, 3],
                                            strides=(tileSize * _strides[0], tileSize * _strides[1], *_strides),
                                            writeable=False)
    # plt.imshow(tiles[0,0]);plt.show()
    # print(tiles[0,0].sum(),tiles[0,0].min())
    tls = tiles.reshape([-1, tileSize, tileSize, 3])
    return tls


class FormsBase(Dataset):
    def __init__(self, vtt=18, htt=14, **kwargs):
        super().__init__()
        self.data = dict()
        self.keys = None
        self.tiles_per_image=(vtt-1)*(htt-1)
        self.vtt=vtt
        self.htt=htt
        self.fl_id_q = Queue(maxsize=1000)
        self.fl_id_s = set()
        self.fl_paths = None

    def __len__(self):
        return len(self.fl_paths)*self.tiles_per_image

    def __getitem__(self, i):
        im_idx=i//self.tiles_per_image
        tl_idx=i%self.tiles_per_image
        if im_idx not in self.fl_id_s:
            if self.fl_id_q.full():
                old_im_id=self.fl_id_q.get()
                self.fl_id_s.remove(old_im_id)
                self.data.pop(old_im_id)
            self.fl_id_q.put(im_idx)
            self.fl_id_s.add(im_idx)
            im=rescaleCrop(self.fl_paths[im_idx])
            tiles=splitImage(im)
            self.data[im_idx]=[tiles[idx] for idx in range(tiles.shape[0])]
        example = dict()
        example["image"]=self.data[im_idx][tl_idx]
        example["class"] = 0
        return example

class FormsTrain(FormsBase):
    def __init__(self, size, keys=None,crop_size=None, coord=False):
        super().__init__()
        root = "data/forms/images"
        with open("data/forms/formstrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.fl_paths = paths
        self.keys = keys

class FormsValidation(FormsBase):
    def __init__(self, size, keys=None,crop_size=None, coord=False):
        super().__init__()
        root = "data/forms/images"
        with open("data/forms/formsvalid.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.fl_paths = paths
        self.keys = keys


class FTileTrain(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/forms/tiles"
        with open("data/forms/tiles_train.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FTileValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/forms/tiles"
        with open("data/forms/tiles_val.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FormsTileTrain(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        self.data = FTileTrain(size=size, keys=keys)
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = [{k:v} for (k,v) in self.data[i].items()]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex


class FormsTileValidation(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        self.data = FTileValidation(size=size, keys=keys)
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.CenterCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = [{k:v} for (k,v) in self.data[i].items()]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex
