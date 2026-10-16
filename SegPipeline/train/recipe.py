"""
The training recipe for SDG datasets, in one place. Used by train_yolo.py, train_seg.py and
tools/profile_train.py. The SDG app's Train page uses the same values (app/server.py PRESETS["strong"]),
and so does the yolo-training repo's "underwater" Albumentations preset.

Speed (measured on the RTX A4000 / i7-12700K PC, tools/profile_train.py): the GPU needs ~0.18 s per
batch of 16; with RandomFog in the pixel augmentations the CPU needed 1.2-2.1 s to prepare one, so the
GPU sat idle ~85% of the time. RandomFog alone cost ~1 s per image it ran on (146 ms per image on
average), 93% of all pixel augmentation time. It is off by default now: the renders already have a
randomized, depth-correct haze (water_fog in the SDG configs). fog=True brings it back.
"""

# Ultralytics' own (label-aware) augmentation: boxes, polygons and keypoints move with the image
HYP = dict(
    hsv_h=0.08, hsv_s=0.9, hsv_v=0.6,       # water colour varies a lot
    degrees=15.0,                           # camera roll
    translate=0.2,
    scale=0.7,                              # 0.3x - 1.7x zoom: far and very close targets
    shear=4.0,
    perspective=0.0008,                     # oblique views
    fliplr=0.5, flipud=0.0,                 # mirrored drawings are still the same class
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.3,                         # segmentation only
)

# schedules
FULL = dict(epochs=150, patience=40, close_mosaic=15)
QUICK = dict(epochs=30, patience=10, close_mosaic=5)     # iteration runs: real-image scores barely move after this
WORKERS = 12                                             # loader processes (20 threads on the training PC)
UNSTRETCH = (1280, 720)                                  # Roboflow exports of the OAK-D frames are stretched to 640x640


def pixel_augmentations(fog=False):
    """Image-only Albumentations transforms (never move pixels, so labels stay valid): make the model
    rely on SHAPE, not on the look of one pool, one renderer or one camera."""
    import albumentations as A

    t = [
        # camera optics / focus / motion
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9)),
            A.MotionBlur(blur_limit=(3, 11)),
            A.MedianBlur(blur_limit=5),
            A.Defocus(radius=(2, 5)),
        ], p=0.4),
        # sensor noise and compression
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.10)),
            A.ISONoise(color_shift=(0.01, 0.06), intensity=(0.1, 0.6)),
        ], p=0.4),
        A.ImageCompression(quality_range=(30, 95), p=0.35),
        A.Downscale(scale_range=(0.35, 0.8), p=0.2),
        # water colour / exposure
        A.RGBShift(r_shift_limit=35, g_shift_limit=35, b_shift_limit=35, p=0.45),
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.45, p=0.6),
        A.RandomGamma(gamma_limit=(60, 160), p=0.3),
        A.CLAHE(clip_limit=(1, 4), p=0.15),
        # colour independence
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Sharpen(alpha=(0.1, 0.4), p=0.1),
    ]
    if fog:                                  # ~1 s per image it runs on: the CPU can't keep the GPU busy
        t.insert(7, A.RandomFog(fog_coef_range=(0.1, 0.45), alpha_coef=0.1, p=0.15))
    return t
