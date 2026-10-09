'''
# External process used by FaceRecognitionTask.py to do the hard work to bypass python GIL and allow multiprocess parallelization.
#
# Recognition models:
#  - 'buffalo_l' (default): InsightFace buffalo_l pack, run with onnxruntime (no insightface package needed):
#      det_10g.onnx (SCRFD-10GF face detector with 5 landmarks) and w600k_r50.onnx (ArcFace ResNet-50, 512-d).
#      Embeddings are L2 normalized: compare them with cosine similarity.
#  - 'dlib': legacy Face Recognition Project (dlib HOG/CNN detector and 128-d ResNet), compared with euclidean distance.
#
# Protocol (stdin/stdout, one item per line):
#   in:  image path, then tiff orientation (or 'video')
#   out: number of faces, one '(top, right, bottom, left)' line per face, then one line per face with the
#        space separated embedding values; or 'image_error'.
'''
import sys
stdout = sys.stdout
sys.stdout = sys.stderr

import glob
import os

import PIL
from PIL import Image, ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

terminate = 'terminate_process'
imgError = "image_error"
ping = "ping"
video = "video"

MODEL_DLIB = 'dlib'
MODEL_BUFFALO_L = 'buffalo_l'

max_files = 2000
processed_files = 0

# Image rotation, when necessary
def rotateImg(img, tiff_orient):
    if tiff_orient == 8 or tiff_orient == 5:
        img = np.rot90(img, 1)
    elif tiff_orient == 3 or tiff_orient == 4:
        img = np.rot90(img, 2)
    elif tiff_orient == 6 or tiff_orient == 7:
        img = np.rot90(img, 3)
    if tiff_orient == 5 or tiff_orient == 7:
        img = np.flipud(img)
    elif tiff_orient == 2 or tiff_orient == 4:
        img = np.fliplr(img)
    return img

# handles Palette images with Transparency expressed in bytes
def convertToRGB(image):
    if image.mode in ("L", "RGB", "P"):
        t = image.info.get("transparency")
        if isinstance(t, bytes):
            image = image.convert('RGBA')
    return image.convert('RGB')


class DlibBackend:
    '''Face Recognition Project (dlib): 128-d embeddings, euclidean distance.'''

    providers = ['dlib']

    def __init__(self, detection_model, up_sampling):
        import face_recognition
        self.fr = face_recognition
        self.detection_model = detection_model
        self.up_sampling = up_sampling

    def process(self, img, max_size, is_video):
        '''Returns (locations, encodings) in original image coordinates.'''
        scale = 1
        upsample = self.up_sampling
        full = img
        h, w = img.shape[:2]
        if not is_video and max(w, h) * 2 > max_size:
            scale = max_size / max(w, h)
            small = Image.fromarray(np.ascontiguousarray(img)).resize((max(1, int(w * scale)), max(1, int(h * scale))),
                                                                       resample=Image.Resampling.BILINEAR)
            img = np.asarray(small)
            upsample = 0
        # Force the array to be C-contiguous in memory for dlib, see https://github.com/sepinf-inc/IPED/issues/2885
        img = np.ascontiguousarray(img)
        locations = self.fr.face_locations(img, number_of_times_to_upsample=upsample, model=self.detection_model)
        if not locations:
            return [], []
        if scale != 1:
            locations = [tuple(int(k / scale) for k in loc) for loc in locations]
        encodings = self.fr.face_encodings(np.ascontiguousarray(full), locations)
        return locations, encodings


# ArcFace 112x112 alignment template (InsightFace face_align.arcface_dst)
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


def add_nvidia_dll_dirs():
    '''onnxruntime-gpu on Windows needs CUDA/cuDNN DLLs; use the nvidia-* pip wheels if installed.'''
    if os.name != 'nt':
        return
    roots = set()
    for p in sys.path:
        if p and os.path.isdir(os.path.join(p, 'nvidia')):
            roots.add(os.path.join(p, 'nvidia'))
    for root in roots:
        for d in glob.glob(os.path.join(root, '*', 'bin')):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
            os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')


def similarity_transform(src, dst):
    '''Umeyama least squares similarity transform (2x3 matrix) mapping src points onto dst points.'''
    n = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = dst_c.T @ src_c / n
    U, S, Vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1
    R = U @ np.diag(d) @ Vt
    var_src = (src_c ** 2).sum() / n
    scale = (S * d).sum() / var_src if var_src > 0 else 1.0
    t = dst_mean - scale * R @ src_mean
    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = scale * R
    M[:, 2] = t
    return M


def nms(dets, thresh):
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


class InsightFaceBackend:
    '''InsightFace buffalo_l with onnxruntime: SCRFD detection + ArcFace w600k_r50, 512-d normalized embeddings.'''

    STRIDES = (8, 16, 32)
    NUM_ANCHORS = 2
    NMS_THRESH = 0.4
    REC_BATCH = 32

    def __init__(self, model_dir, device='auto', det_thresh=0.5, cpu_threads=0, min_face=0):
        det_path = os.path.join(model_dir, 'det_10g.onnx')
        rec_path = os.path.join(model_dir, 'w600k_r50.onnx')
        for p in (det_path, rec_path):
            if not os.path.isfile(p):
                raise FileNotFoundError('InsightFace model file not found: ' + p)
        if device != 'cpu':
            add_nvidia_dll_dirs()
        import onnxruntime as ort
        available = ort.get_available_providers()
        if device == 'cpu':
            providers = ['CPUExecutionProvider']
        else:
            providers = [p for p in ('CUDAExecutionProvider', 'DmlExecutionProvider') if p in available]
            providers.append('CPUExecutionProvider')
        options = ort.SessionOptions()
        if cpu_threads > 0:
            options.intra_op_num_threads = cpu_threads
        options.log_severity_level = 3
        self.det = ort.InferenceSession(det_path, sess_options=options, providers=providers)
        self.rec = ort.InferenceSession(rec_path, sess_options=options, providers=providers)
        self.det_input = self.det.get_inputs()[0].name
        self.rec_input = self.rec.get_inputs()[0].name
        self.det_thresh = det_thresh
        self.min_face = min_face
        self.providers = self.det.get_providers()
        self._anchor_cache = {}

    def _anchors(self, size, stride):
        key = (size, stride)
        centers = self._anchor_cache.get(key)
        if centers is None:
            ys, xs = np.mgrid[:size, :size]
            centers = (np.stack([xs, ys], axis=-1).astype(np.float32) * stride).reshape(-1, 2)
            centers = np.repeat(centers, self.NUM_ANCHORS, axis=0)
            self._anchor_cache[key] = centers
        return centers

    def detect(self, img, max_size):
        '''Returns boxes (n, 5: x1, y1, x2, y2, score) and landmarks (n, 5, 2) in image coordinates.'''
        h, w = img.shape[:2]
        # square canvas with the image letterboxed at top-left, as in InsightFace SCRFD. Few fixed sizes:
        # each new input shape costs seconds of cuDNN autotuning on GPU
        side = max(h, w)
        max_canvas = max(320, max_size - max_size % 32)
        canvas = min(max_canvas, 320 if side <= 320 else 640 if side <= 640 else max_canvas)
        scale = canvas / max(h, w)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = Image.fromarray(np.ascontiguousarray(img)).resize((nw, nh), resample=Image.Resampling.BILINEAR)
        blob = np.zeros((canvas, canvas, 3), dtype=np.float32)
        blob[:nh, :nw] = np.asarray(resized, dtype=np.float32)
        blob = ((blob - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outs = self.det.run(None, {self.det_input: blob})
        fmc = len(self.STRIDES)
        boxes, kpss = [], []
        for idx, stride in enumerate(self.STRIDES):
            scores = outs[idx].reshape(-1)
            pos = np.where(scores >= self.det_thresh)[0]
            if pos.size == 0:
                continue
            bbox = outs[idx + fmc].reshape(-1, 4)[pos] * stride
            kps = outs[idx + fmc * 2].reshape(-1, 10)[pos] * stride
            c = self._anchors(canvas // stride, stride)[pos]
            boxes.append(np.stack([c[:, 0] - bbox[:, 0], c[:, 1] - bbox[:, 1], c[:, 0] + bbox[:, 2],
                                   c[:, 1] + bbox[:, 3], scores[pos]], axis=-1))
            kpss.append(kps.reshape(-1, 5, 2) + c[:, None, :])
        if not boxes:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        boxes = np.concatenate(boxes)
        kpss = np.concatenate(kpss)
        boxes[:, :4] /= scale
        kpss /= scale
        keep = nms(boxes, self.NMS_THRESH)
        boxes, kpss = boxes[keep], kpss[keep]
        if self.min_face > 0:
            big = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) >= self.min_face
            boxes, kpss = boxes[big], kpss[big]
        return boxes, kpss

    @staticmethod
    def align(pil_img, kps):
        '''112x112 ArcFace crop: PIL samples the inverse of the similarity transform.'''
        M = similarity_transform(kps.astype(np.float64), ARCFACE_DST)
        inv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))
        crop = pil_img.transform((112, 112), Image.Transform.AFFINE, data=tuple(inv[:2].reshape(-1)),
                                 resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
        return np.asarray(crop, dtype=np.float32)

    def embed(self, crops):
        out = []
        for i in range(0, len(crops), self.REC_BATCH):
            batch = np.stack(crops[i:i + self.REC_BATCH])
            batch = ((batch - 127.5) / 127.5).transpose(0, 3, 1, 2)
            feats = self.rec.run(None, {self.rec_input: batch})[0].astype(np.float64)
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            out.extend(feats / np.where(norms > 0, norms, 1))
        return out

    def process(self, img, max_size, is_video):
        boxes, kpss = self.detect(img, max_size)
        if len(boxes) == 0:
            return [], []
        h, w = img.shape[:2]
        pil_img = Image.fromarray(np.ascontiguousarray(img))
        locations = []
        for b in boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
            # same (top, right, bottom, left) convention of face_recognition
            locations.append((max(0, y1), min(w - 1, x2), min(h - 1, y2), max(0, x1)))
        encodings = self.embed([self.align(pil_img, k) for k in kpss])
        return locations, encodings


def create_backend(model, detection_model, up_sampling, model_dir, device, det_thresh, cpu_threads, min_face):
    if model == MODEL_DLIB:
        return DlibBackend(detection_model, up_sampling)
    if model == MODEL_BUFFALO_L:
        return InsightFaceBackend(model_dir, device, det_thresh, cpu_threads, min_face)
    raise ValueError('Unknown face recognition model: ' + model)


def format_encoding(encoding):
    return ' '.join('%.7g' % v for v in encoding)


'''
Main function of external process which will detect and encode faces.
It is executed out of process to workaroung python GIL bottleneck.
Multiprocessing module does not work with jep-3.9.1.
'''
def main():
    global processed_files, max_files
    max_size = int(sys.argv[1])
    detection_model = sys.argv[2]
    up_sampling = int(sys.argv[3])
    model = sys.argv[4] if len(sys.argv) > 4 else MODEL_DLIB
    model_dir = sys.argv[5] if len(sys.argv) > 5 else ''
    device = sys.argv[6] if len(sys.argv) > 6 else 'auto'
    det_thresh = float(sys.argv[7]) if len(sys.argv) > 7 else 0.5
    cpu_threads = int(sys.argv[8]) if len(sys.argv) > 8 else 0
    min_face = int(sys.argv[9]) if len(sys.argv) > 9 else 0

    # load models before answering the first ping, so setup errors show up at process creation
    backend = create_backend(model, detection_model, up_sampling, model_dir, device, det_thresh, cpu_threads, min_face)
    if model != MODEL_DLIB:
        # no known resource leaks and model loading is slower: restart less often
        max_files = 50000
        print('Face recognition model ' + model + ' loaded, providers: ' + str(backend.providers), file=sys.stderr, flush=True)

    while True:
        if processed_files >= max_files:
            break

        line = input()
        if line == terminate:
            break
        if line == ping:
            print(ping, file=stdout, flush=True)
            continue

        processed_files += 1

        code = input()
        if code == video:
            isVideo = True
            tiff_orient = 1
        else:
            isVideo = False
            tiff_orient = int(code)

        try:
            img = PIL.Image.open(line)
            img = convertToRGB(img)
            img = rotateImg(np.asarray(img), tiff_orient)
        except Exception:
            print(imgError, file=stdout, flush=True)
            continue

        try:
            locations, encodings = backend.process(img, max_size, isVideo)
        except Exception as e:
            print('Error processing ' + line + ': ' + repr(e), file=sys.stderr, flush=True)
            print(imgError, file=stdout, flush=True)
            continue

        print(str(len(locations)), file=stdout, flush=True)
        for loc in locations:
            print(str(tuple(int(k) for k in loc)), file=stdout, flush=True)
        for enc in encodings:
            print(format_encoding(enc), file=stdout, flush=True)
    return

if __name__ == "__main__":
     main()
