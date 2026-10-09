"""
Tests of scripts/tasks/FaceRecognitionProcess.py (buffalo_l backend).

Run with the IPED python, which has numpy, pillow and onnxruntime installed:

    python -m pytest iped-app/src/test/python -q

or without pytest:

    python iped-app/src/test/python/test_face_recognition_process.py

Tests with the real models run when the buffalo_l files are found in IPED_FACE_MODEL_DIR or
~/.insightface/models/buffalo_l, and face photos are in IPED_FACE_TESTDATA (optional).
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'resources', 'scripts', 'tasks'))
sys.path.insert(0, SCRIPTS)
import FaceRecognitionProcess as fp  # noqa: E402

MODEL_DIR = os.environ.get('IPED_FACE_MODEL_DIR',
                           os.path.join(os.path.expanduser('~'), '.insightface', 'models', 'buffalo_l'))
FACES = os.environ.get('IPED_FACE_TESTDATA', '')
HAS_MODEL = os.path.isfile(os.path.join(MODEL_DIR, 'w600k_r50.onnx'))


def skip(reason):
    try:
        import pytest
        pytest.skip(reason)
    except ImportError:
        print('skipped: ' + reason)
        raise _Skip()


class _Skip(Exception):
    pass


def test_similarity_transform_recovers_rotation_scale_translation():
    angle = np.deg2rad(20)
    R = 1.7 * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    src = np.array([[10, 20], [50, 22], [30, 40], [15, 60], [45, 62]], dtype=np.float64)
    dst = src @ R.T + np.array([5, -3])
    M = fp.similarity_transform(src, dst)
    assert np.allclose(M[:, :2], R, atol=1e-6)
    assert np.allclose(M[:, 2], [5, -3], atol=1e-6)


def test_nms_keeps_best_of_overlapping_boxes():
    dets = np.array([[0, 0, 100, 100, 0.9], [5, 5, 105, 105, 0.8], [200, 200, 260, 260, 0.7]], dtype=np.float32)
    assert sorted(fp.nms(dets, 0.4)) == [0, 2]


def test_rotate_and_rgb_helpers_kept_for_age_estimation():
    img = np.arange(6).reshape(2, 3)
    assert fp.rotateImg(img, 6).shape == (3, 2)
    p = Image.new('P', (4, 4))
    assert fp.convertToRGB(p).mode == 'RGB'


def _backend():
    if not HAS_MODEL:
        skip('buffalo_l model not found in ' + MODEL_DIR)
    return fp.InsightFaceBackend(MODEL_DIR)


def test_no_face_in_blank_image():
    backend = _backend()
    locations, encodings = backend.process(np.full((480, 640, 3), 127, np.uint8), 1024, False)
    assert locations == [] and encodings == []


def test_same_person_is_more_similar_than_different_people():
    if not FACES or not os.path.isdir(FACES):
        skip('set IPED_FACE_TESTDATA with pessoaA_1/2.jpg and pessoaB_1/2.jpg')
    backend = _backend()
    emb = {}
    for name in ('pessoaA_1', 'pessoaA_2', 'pessoaB_1', 'pessoaB_2'):
        img = np.asarray(fp.convertToRGB(Image.open(os.path.join(FACES, name + '.jpg'))))
        locations, encodings = backend.process(img, 1024, False)
        assert len(locations) == 1, name
        top, right, bottom, left = locations[0]
        assert 0 <= left < right < img.shape[1] and 0 <= top < bottom < img.shape[0]
        assert encodings[0].shape == (512,)
        assert abs(np.linalg.norm(encodings[0]) - 1) < 1e-5
        emb[name] = encodings[0]
    same = [emb['pessoaA_1'] @ emb['pessoaA_2'], emb['pessoaB_1'] @ emb['pessoaB_2']]
    diff = [emb[a] @ emb[b] for a in ('pessoaA_1', 'pessoaA_2') for b in ('pessoaB_1', 'pessoaB_2')]
    print('same', same, 'different', diff)
    assert min(same) > 0.5
    assert max(diff) < 0.2


def test_external_process_protocol(tmp_path=None):
    if not HAS_MODEL:
        skip('buffalo_l model not found')
    import tempfile
    folder = str(tmp_path) if tmp_path else tempfile.mkdtemp()
    blank = os.path.join(folder, 'blank.png')
    Image.new('RGB', (320, 240), (90, 90, 90)).save(blank)
    images = [blank]
    if FACES and os.path.isfile(os.path.join(FACES, 'pessoaA_1.jpg')):
        images.append(os.path.join(FACES, 'pessoaA_1.jpg'))
    proc = subprocess.Popen([sys.executable, os.path.join(SCRIPTS, 'FaceRecognitionProcess.py'), '1024', 'hog', '1',
                             'buffalo_l', MODEL_DIR, 'auto', '0.5', '0', '0'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    try:
        print(fp.ping, file=proc.stdin, flush=True)
        assert proc.stdout.readline().strip() == fp.ping
        print(os.path.join(folder, 'missing.jpg'), file=proc.stdin, flush=True)
        print('1', file=proc.stdin, flush=True)
        assert proc.stdout.readline().strip() == fp.imgError
        for path in images:
            print(path, file=proc.stdin, flush=True)
            print('1', file=proc.stdin, flush=True)
            n = int(proc.stdout.readline())
            if path == blank:
                assert n == 0
                continue
            assert n == 1
            loc = eval(proc.stdout.readline())
            assert len(loc) == 4
            values = proc.stdout.readline().split()
            assert len(values) == 512
        print(fp.terminate, file=proc.stdin, flush=True)
        proc.wait(30)
    finally:
        if proc.poll() is None:
            proc.kill()


if __name__ == '__main__':
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('ok   ' + name)
            except _Skip:
                print('skip ' + name)
            except Exception as e:
                failed += 1
                print('FAIL ' + name + ': ' + repr(e))
    sys.exit(1 if failed else 0)
