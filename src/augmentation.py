import tensorflow as tf
import math
import tensorflow_addons as tfa

def augment_batch_old(img_batch, y_batch, p=0.1):
    """
    Applies random_augment_pair to each (img_batch[i], y_batch[i]).
    """

    # Enforce known rank-4 shape on the entire batch if possible
    # If you know the channel count is 3, you can do:
    img_batch = tf.ensure_shape(img_batch, [None, None, None, 3])
    y_batch = tf.ensure_shape(y_batch, [None, None, None, 3])

    def _augment_one_pair(inputs):
        img, target = inputs
        # Now each is rank-3 [H, W, 3]
        return random_augment_pair(img, target, p=p)

    augmented_img, augmented_target = tf.map_fn(
        _augment_one_pair,
        (img_batch, y_batch),
        dtype=(img_batch.dtype, y_batch.dtype)
    )
    
    return augmented_img, augmented_target

def augment_batch(img_batch, y_batch, p=0.1):
    """
    Applies random_augment_pair to each (img_batch[i], y_batch[i]).
    Accepts inputs of either rank-3 (a single image) or rank-4 (a batch of images),
    and returns outputs with the same rank as the input.
    """
    # Check if the input is rank-3 (i.e. a single image)
    is_single = False
    if len(img_batch.shape) == 3:
        is_single = True
        img_batch = tf.expand_dims(img_batch, axis=0)
    if len(y_batch.shape) == 3:
        y_batch = tf.expand_dims(y_batch, axis=0)

    # Enforce known shape with 3 channels
    img_batch = tf.ensure_shape(img_batch, [None, None, None, 3])
    y_batch = tf.ensure_shape(y_batch, [None, None, None, 3])

    def _augment_one_pair(inputs):
        img, target = inputs
        # Each image is now rank-3 [H, W, 3]
        return random_augment_pair(img, target, p=p)

    augmented_img, augmented_target = tf.map_fn(
        _augment_one_pair,
        (img_batch, y_batch),
        dtype=(img_batch.dtype, y_batch.dtype)
    )

    # If the original input was rank-3, remove the batch dimension before returning
    if is_single:
        augmented_img = tf.squeeze(augmented_img, axis=0)
        augmented_target = tf.squeeze(augmented_target, axis=0)

    return augmented_img, augmented_target


def random_augment_pair(img, target, p=0.1):
    """
    Apply random augmentations to (img, target) with probability p=0.1 each.
    Both img & target are rank-3 Tensors [H, W, 3], float32 in [0..1].
    """
    img = tf.cast(img, tf.float32)
    target = tf.cast(target, tf.float32)
    
    # Ensure shapes are rank-3 with 3 channels
    img = tf.ensure_shape(img, [None, None, 3])
    target = tf.ensure_shape(target, [None, None, 3])

    # 1) Random horizontal flip
    flip_prob = tf.random.uniform(())
    def flip_fn():
        return (tf.image.flip_left_right(img),
                tf.image.flip_left_right(target))
    img, target = tf.cond(
        flip_prob < p, flip_fn, lambda: (img, target))
    
    # 2) Random shifting
    shift_prob = tf.random.uniform(())
    def shift_fn():
        return random_shift(img, target)
    img, target = tf.cond(
        shift_prob < p, shift_fn, lambda: (img, target))

    # 3) Slight random rotation in [-3°, 3°]
    rot_prob = tf.random.uniform(())
    def rotate_fn():
        angle_deg = tf.random.uniform((), -3.0, 3.0)
        angle_rad = angle_deg * math.pi / 180.0
        return (tfa.image.rotate(img, angle_rad, interpolation='BILINEAR'),
                tfa.image.rotate(target, angle_rad, interpolation='BILINEAR'))
    img, target = tf.cond(
        rot_prob < p, rotate_fn, lambda: (img, target))

    # 4) Slight random zoom in [0.9..1.1]
    zoom_prob = tf.random.uniform(())
    def zoom_fn():
        scale = tf.random.uniform((), 0.9, 1.1)
        return random_zoom(img, target, scale)
    img, target = tf.cond(
        zoom_prob < p, zoom_fn, lambda: (img, target))

    # 5) Random occlusion (cutout)
    occ_prob = tf.random.uniform(())
    def occ_fn():
        return random_occlusion(img, target)
    img, target = tf.cond(
        occ_prob < p, occ_fn, lambda: (img, target))

    # 6) Random brightness shift in [-0.3, 0.3]
    bright_prob = tf.random.uniform(())
    def bright_fn():
        brightness_delta = tf.random.uniform((), -0.3, 0.3)
        img_b = tf.clip_by_value(img + brightness_delta, 0.0, 1.0)
        tgt_b = tf.clip_by_value(target + brightness_delta, 0.0, 1.0)
        return img_b, tgt_b
    img, target = tf.cond(
        bright_prob < p, bright_fn, lambda: (img, target))
    
    # 7) Random Grayscale 
    gray_prob = tf.random.uniform(())
    def gray_fn():
        # Convert to grayscale => shape [H, W, 1]
        g_img = tf.image.rgb_to_grayscale(img)
        g_tgt = tf.image.rgb_to_grayscale(target)
        # Expand back to 3 channels
        g_img = tf.tile(g_img, [1, 1, 3])
        g_tgt = tf.tile(g_tgt, [1, 1, 3])
        return g_img, g_tgt
    img, target = tf.cond(
        gray_prob < p, gray_fn, lambda: (img, target))
    
    # 8) Random downsampling augmentation
    # For downsampling we want target to be unchanged to allow image restoration to be learned
    ds_prob = tf.random.uniform(())
    def ds_fn():
        # Get the original image shape
        orig_shape = tf.shape(img)
        # Compute new dimensions as half the original
        new_h = orig_shape[0] // 2
        new_w = orig_shape[1] // 2
        # Downsample to half size
        img_down = tf.image.resize(img, [new_h, new_w], method='area')
        target_down = tf.image.resize(target, [new_h, new_w], method='area')
        # Upsample back to original size
        img_up = tf.image.resize(img_down, [orig_shape[0], orig_shape[1]], method='bicubic')
        target_up = tf.image.resize(target_down, [orig_shape[0], orig_shape[1]], method='bicubic')
        return img_up, target_up
    #Notice we don't overwrite the target,  we want to learn to recover from downsampling
    img, _ = tf.cond(ds_prob < p, ds_fn, lambda: (img, target))

    img = tf.cast(img, tf.float16)
    target = tf.cast(target, tf.float16)

    return img, target

def random_zoom(img, target, scale):
    """
    Zoom by factor `scale`:
      - scale < 1 => 'zoom in' (center-crop)
      - scale > 1 => 'zoom out' (center-pad)
    Output shape is the same as input shape [H, W, 3].
    """
    img = tf.ensure_shape(img, [None, None, 3])
    target = tf.ensure_shape(target, [None, None, 3])

    shape = tf.shape(img)
    h, w = shape[0], shape[1]

    new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)

    # Resize both
    img_resized = tf.image.resize(img, [new_h, new_w], method='bilinear')
    target_resized = tf.image.resize(target, [new_h, new_w], method='bilinear')

    # Compute offsets using tf ops
    offset_h = tf.math.floordiv((h - new_h), 2)  # integer division in TF
    offset_w = tf.math.floordiv((w - new_w), 2)

    # If offset is negative, that means new_h > h (zoom out).
    # We'll handle that with pad_to_bounding_box and then a final crop.
    offset_h_pos = tf.maximum(offset_h, 0)
    offset_w_pos = tf.maximum(offset_w, 0)
    offset_h_neg = tf.abs(tf.minimum(offset_h, 0))
    offset_w_neg = tf.abs(tf.minimum(offset_w, 0))

    # Pad
    img_padded = tf.image.pad_to_bounding_box(
        img_resized,
        offset_h_pos,
        offset_w_pos,
        tf.maximum(h, new_h),
        tf.maximum(w, new_w)
    )
    target_padded = tf.image.pad_to_bounding_box(
        target_resized,
        offset_h_pos,
        offset_w_pos,
        tf.maximum(h, new_h),
        tf.maximum(w, new_w)
    )

    # Crop back to [h, w]
    img_final = tf.image.crop_to_bounding_box(
        img_padded,
        offset_h_neg,
        offset_w_neg,
        h,
        w
    )
    target_final = tf.image.crop_to_bounding_box(
        target_padded,
        offset_h_neg,
        offset_w_neg,
        h,
        w
    )

    # Ensure shape
    img_final = tf.ensure_shape(img_final, [None, None, 3])
    target_final = tf.ensure_shape(target_final, [None, None, 3])

    return img_final, target_final



def random_occlusion(img, target):
    """
    Random cutout: pick a rectangle up to ~10% area, fill with 0.0.
    """
    img = tf.ensure_shape(img, [None, None, 3])
    target = tf.ensure_shape(target, [None, None, 3])

    h = tf.shape(img)[0]
    w = tf.shape(img)[1]

    max_box_h = tf.cast(tf.round(tf.cast(h, tf.float32) * 0.25), tf.int32)
    max_box_w = tf.cast(tf.round(tf.cast(w, tf.float32) * 0.25), tf.int32)

    box_h = tf.random.uniform((), 1, max_box_h+1, dtype=tf.int32)
    box_w = tf.random.uniform((), 1, max_box_w+1, dtype=tf.int32)

    if box_h > box_w:
        box_w = box_h
    else:
        box_h = box_w

    y0 = tf.random.uniform((), 0, h - box_h + 1, dtype=tf.int32)
    x0 = tf.random.uniform((), 0, w - box_w + 1, dtype=tf.int32)

    # Build mask
    mask_2d = tf.ones([h, w], dtype=tf.float32)

    coords_y, coords_x = tf.meshgrid(
        tf.range(y0, y0+box_h),
        tf.range(x0, x0+box_w),
        indexing='ij'
    )
    coords = tf.stack([tf.reshape(coords_y, [-1]),
                       tf.reshape(coords_x, [-1])], axis=-1)

    mask_2d = tf.tensor_scatter_nd_update(
        mask_2d, coords, tf.zeros([box_h * box_w], dtype=tf.float32)
    )
    mask_3d = tf.expand_dims(mask_2d, axis=-1)

    img_occluded = img * mask_3d
    #target_occluded = target * mask_3d

    return img_occluded, target #target_occluded

def random_shift(img, target, max_shift=5):
    """
    Randomly shifts the image and target up, down, left, or right.
    Pads with black fill (0.0) for the uncovered areas.
    """
    img = tf.ensure_shape(img, [None, None, 3])
    target = tf.ensure_shape(target, [None, None, 3])
    
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]

    # Generate random shift values for height and width
    shift_h = tf.random.uniform((), -max_shift, max_shift + 1, dtype=tf.int32)
    shift_w = tf.random.uniform((), -max_shift, max_shift + 1, dtype=tf.int32)

    # Apply shifts with padding and cropping
    img_shifted = tf.image.pad_to_bounding_box(img, max_shift, max_shift, h + 2 * max_shift, w + 2 * max_shift)
    target_shifted = tf.image.pad_to_bounding_box(target, max_shift, max_shift, h + 2 * max_shift, w + 2 * max_shift)

    img_shifted = tf.image.crop_to_bounding_box(img_shifted, max_shift - shift_h, max_shift - shift_w, h, w)
    target_shifted = tf.image.crop_to_bounding_box(target_shifted, max_shift - shift_h, max_shift - shift_w, h, w)

    # Ensure shapes
    img_shifted = tf.ensure_shape(img_shifted, [None, None, 3])
    target_shifted = tf.ensure_shape(target_shifted, [None, None, 3])

    return img_shifted, target_shifted
