import os
import numpy as np
import tensorflow as tf
import cv2  # If you need to visualize or save images easily
from face_analysis import FaceAnalysis
from inswapper import INSwapper
from aei_net import get_model
from tensorflow.keras import optimizers
from emap import emap
from tensorflow_addons.optimizers import AdamW
from AEIGANModel import AEIGANModel
from discriminator import MinimalPatchGAN 
from super_resolution import SuperResModel 
from skimage import transform as trans
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
import uuid
from gpu_memory import limit_gpu_memory


cv2.setUseOptimized(True)
cv2.ocl.setUseOpenCL(True)

from onnx_upsampler import Upsampler_Remote
upsampler=Upsampler_Remote()

IMAGE_SIZE = 256
CHANNELS = 3
BATCH_SIZE = 32
NUM_FEATURES = 512
NUM_BLOCKS=2

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

class AEINETSwapper():
    """
    # Example Usage
    ```python
    import cv2
    from PIL import Image
    from inference import AEINETSwapper
    face_swapper = AEINETSwapper(model_paths=(f"./models/MODEL_256x256_v5_BLOCKS2_latest", f""))
    _, into_image = face_swapper.read_and_scale('./samples/x.jpg', 256, True)
    results = face_swapper.process_image(into_image, faces_directory='./samples/')
    for i, img in enumerate(results):
      img.save(f"./output/{i}.png")
    ```
    """
    def __init__(self, model_paths, batch_size=32, input_size=(IMAGE_SIZE, IMAGE_SIZE), with_super_resolution=False, use_emap=True):
        self.face_swapper, self.trainable_model = self.load_model(model_paths, with_super_resolution=with_super_resolution, use_emap=use_emap)

        self.face_analyser = FaceAnalysis()
        self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))

        self.input_mean = 0.0
        self.input_std = 255.0
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = input_size if not with_super_resolution else (512, 512)
        self.emap = emap
        self.use_emap = use_emap
        self.with_super_resolution = with_super_resolution
        

    def load_model(self, model_paths, num_blocks=NUM_BLOCKS, with_super_resolution=False, use_emap=True):
        """
        Load your pre-trained Keras model from disk.
        Adjust this as needed for your particular setup.
        """
        self.model_weights, self.superres_model_weights = model_paths
        with tf.device('/GPU:0'):
            # Load aei-net model generator and discriminator
            discriminator = MinimalPatchGAN(ndf=64, num_layers=3)
            #discriminator.trainable = False
            model = AEIGANModel(
                generator=get_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), num_blocks=NUM_BLOCKS, freeze_all_except_deconv=False, with_super_resolution=with_super_resolution),
                discriminator=discriminator,
                lambda_recon=150.0, lambda_adv=0.5, lambda_id=45.0,
                use_emap=use_emap)
            model.compile(
                g_optimizer=AdamW(learning_rate=1e-4, beta_1=0, beta_2=0.999, weight_decay=1e-4), 
                d_optimizer=AdamW(learning_rate=1e-6, beta_1=0, beta_2=0.999, weight_decay=1e-4), 
                run_eagerly=True,
                split_batch_by=2
            )

            # Load old weights if required
            if self.model_weights and os.path.isdir(self.model_weights):
                print(f"###### Using model weights: {self.model_weights}")
                model.load_weights(self.model_weights)
                model.g_optimizer=AdamW(learning_rate=1e-4, beta_1=0, beta_2=0.999, weight_decay=1e-4) 
                model.d_optimizer=AdamW(learning_rate=1e-6, beta_1=0, beta_2=0.999, weight_decay=1e-4)
            else:
                print(f"###### No model weights found at: {self.model_weights}")
        
        # Prime the models with fake data    
        dummy_img = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS], dtype=tf.float32)
        dummy_embed = tf.zeros([1, NUM_FEATURES], dtype=tf.float32)
        model((dummy_img, dummy_embed), training=False)
        
        self.model = model
        
        return model.generator, model
    
    def reload(self):
        # Load old weights if required
        if self.model_weights and os.path.isdir(self.model_weights):
            print(f"###### Using model weights: {self.model_weights}")
            self.model.load_weights(self.model_weights)
            self.model.g_optimizer=AdamW(learning_rate=1e-4, beta_1=0, beta_2=0.999, weight_decay=1e-4) 
            self.model.d_optimizer=AdamW(learning_rate=1e-6, beta_1=0, beta_2=0.999, weight_decay=1e-4)
        else:
            print(f"###### No model weights found at: {self.model_weights}")

    def ensure_rgb(self, image):
        if image.shape[-1] == 4:  # If the image has an alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image

    def read_and_scale(self, img_path, WxH=IMAGE_SIZE, swapRGB=False):
        if isinstance(img_path, np.ndarray):
            img = img_path
        else:
            img = cv2.imread(img_path)

        img = self.ensure_rgb(img)

        if swapRGB:
            img = img[:,:,::-1]
        return self.scale_image(img, WxH)

    def scale_image(self, img, WxH=IMAGE_SIZE):
        height, width = img.shape[:2]

        faces = self.face_analyser.get(img)

        scaling_factor = 1.0
        for bbox in [face.bbox for face in faces]:
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if width > WxH or height > WxH:
                scale =(WxH * 1.0) / max(width, height)
                scaling_factor = min(scaling_factor, scale)

        if scaling_factor < 1.0:
            new_size = (int(img.shape[1] * scaling_factor), int(img.shape[0] * scaling_factor))
            return scaling_factor, cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        else:
            return scaling_factor, img
        
    def extract_face(self, image, face_analyser):
        frame = np.array(image) if isinstance(image, Image.Image) else image
        faces = sorted(self.face_analyser.get(frame), key=lambda x: x.bbox[0])

        if not faces:
            return None  # No face detected

        # Get the first detected face (leftmost one)
        face = faces[0]
        x1, y1, x2, y2 = map(int, face.bbox)  # Ensure bbox coordinates are integers

        # Crop the face region
        face_crop = frame[y1:y2, x1:x2]

        return face_crop


    def process_image(self, image, faces_directory='/app/faces/'):  
        for filename in os.listdir(faces_directory):
            if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                image_path = os.path.join(faces_directory, filename)
                _, source_image = self.read_and_scale(image_path)
                #source_image = cv2.imread(image_path)
                source_face = min(self.face_analyser.get(source_image), key=lambda x: x.bbox[0])
                #print(f'Face Dimensions: {(source_face.bbox[2] - source_face.bbox[0])}x{(source_face.bbox[3] - source_face.bbox[1])}')
                frame = np.array(image)
                faces = sorted(self.face_analyser.get(np.array(frame)), key=lambda x: x.bbox[0])
            
                if faces:
                    for i, face in enumerate(faces):
                        frame = self.get(frame, face, source_face, paste_back=True)
                
                result = frame
                yield Image.fromarray(result)

    #def process_batch_old(self, frames, source_image):
    #    out = []
    #    source_face = min(self.face_analyser.get(source_image), key=lambda x: x.bbox[0])
    #    for i, frame in enumerate(frames):
    #        faces = sorted(self.face_analyser.get(frame), key=lambda x: x.bbox[0])
    #        
    #        if faces:
    #            for i, face in enumerate(faces):
    #                frame = self.get(frame, face, source_face, paste_back=True)
    #        else:
    #            print(f"No face in frame {i}")
    #    
    #        out.append(frame)
    #
    #    return out

    def get_average_face(self, images):
        class local_face(object):
            pass

        #Average the inputs
        if isinstance(images, list):           
            faces = []
            primary_face = local_face()
            primary_face.normed_embedding = None
            for i, image in enumerate(images):
                face = sorted(self.face_analyser.get(image), key=lambda x: x.bbox[0])
                if not face:
                    print(f"No face found @ index={i}")
                    return images[0]
                faces.append(face[0])
            
            # Compute the average of the normed_embedding for all detected faces
            avg_embedding = np.mean([face.normed_embedding for face in faces], axis=0)
            primary_face.normed_embedding = avg_embedding

            return primary_face
        else:
            # Detect exactly one face in source_image (pick the left-most, if multiple)
            source_faces = sorted(self.face_analyser.get(images), key=lambda x: x.bbox[0])

            return source_faces[0]


    def process_batch(self, frames, source_face, batch_size=32, upsample=False):
        # This will hold the final swapped (or original) frames in correct order
        final_frames = [None] * len(frames)

        if not source_face:
            print("No face found in source image. Returning frames unchanged.")
            return frames

        # Accumulate all frames (and their first detected face, if any) into a list
        frames_with_faces = []
        for i, frame in enumerate(frames):
            detected_faces = sorted(self.face_analyser.get(frame), key=lambda x: x.bbox[0])
            if detected_faces:
                # Store the first detected face
                face = detected_faces[0]
                frames_with_faces.append((i, frame, face))
            else:
                # No face in this frame => output should remain the original
                final_frames[i] = frame
                print(f"No face detected in frame {i}")

        # Now process frames_with_faces in batches of up to 32
        for start_idx in range(0, len(frames_with_faces), batch_size):
            batch = frames_with_faces[start_idx : start_idx + batch_size]

            # Pull out indices, frames, and faces separately
            batch_indices = [item[0] for item in batch]
            batch_frames  = [item[1] for item in batch]
            batch_faces   = [item[2] for item in batch]

            # Call your existing get(...) function once per batch
            #   imgs         -> batch_frames
            #   target_faces -> batch_faces
            #   source_faces -> single face object (this will get expanded internally)
            swapped_batch = self.get(batch_frames, batch_faces, source_face, paste_back=True ,upsample=upsample)

            # Write them back to final_frames in the correct positions
            for idx, swapped_frame in zip(batch_indices, swapped_batch):
                final_frames[idx] = swapped_frame

        return final_frames

    #https://github.com/postworthy/FaceSwapPipeline/blob/78b2d0efb0d4a969db7407640db55d48c7512801/inswapper.py#L85
    def get(self, imgs, target_faces, source_faces, paste_back=True, upsample=False):
        # Process batch inputs
        blobs = []
        latents = []
        Ms = []
        aimgs = []
        is_single=False
        return_count = self.batch_size
        

        if not isinstance(imgs, list):
            imgs = [imgs] 
            is_single=True
        else:
            return_count = len(imgs)

        if not isinstance(target_faces, list):
            target_faces = [target_faces] * len(imgs)
        if not isinstance(source_faces, list):
            source_faces = [source_faces] * len(imgs)

        #print(f"imgs shape: {imgs[0].shape}")

        while len(imgs) < self.batch_size:
            imgs.append(imgs[-1])

        while len(target_faces) < len(imgs):
            target_faces.append(target_faces[-1])

        while len(source_faces) < len(imgs):
            source_faces.append(source_faces[-1])

        for idx in range(len(imgs)):
            img = imgs[idx]
            target_face = target_faces[idx]
            source_face = source_faces[idx]

            aimg, M = self.norm_crop2(img, target_face.kps, self.output_size[0])
            
            blob = cv2.dnn.blobFromImage(
                aimg,
                1.0 / self.input_std,
                (IMAGE_SIZE, IMAGE_SIZE),
                (self.input_mean, self.input_mean, self.input_mean),
                swapRB=False
            )
            
            blob = tf.transpose(blob, perm=[0, 2, 3, 1])
            latent = source_face.normed_embedding.reshape((1, -1))
            latent = np.dot(latent, self.emap)
            latent /= np.linalg.norm(latent)

            aimgs.append(aimg)
            blobs.append(blob)
            latents.append(latent)
            Ms.append(M)


        # Stack blobs and latents
        blobs = np.vstack(blobs)
        latents = np.vstack(latents)

        # Run the model
        preds, preds_small, _ = self.face_swapper([blobs, latents])

        #If you want to simply override the models upsampled version
        #preds = cv2.resize(preds_small, (512, 512), interpolation=cv2.INTER_AREA)
        #preds = tf.image.resize(preds_small, size=(512,512), method=tf.image.ResizeMethod.BILINEAR)

        if upsample: #and not self.with_super_resolution:
            preds = upsampler.upsample(preds, downscale=False if self.with_super_resolution else True)

        
        img_is_small = [max(img.shape[:2]) <= 256 for img in imgs]
        chosen_preds = [
            ps if small else pl
            for ps, pl, small in zip(preds_small, preds, img_is_small)
        ]

        #Save for debug purposes
        #self.save_preds(preds[:,:,:,::-1], output_dir="output")
        #self.save_preds(preds_small[:,:,:,::-1], output_dir="output")

        # Postprocess the outputs
        fake_images = []
        #for i in range(len(preds)):
        #    pred = preds[i]
        for i in range(len(chosen_preds)):
            pred = chosen_preds[i]
            #img_fake = tf.transpose(pred, perm=[1, 2, 0])
            img_fake = pred
            bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)
            if not paste_back:
                fake_images.append((bgr_fake, Ms[i]))
            else:
                img = imgs[i]
                aimg = aimgs[i]
                M = Ms[i]

                target_img = img
                fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
                fake_diff = np.abs(fake_diff).mean(axis=2)
                fake_diff[:2,:] = 0
                fake_diff[-2:,:] = 0
                fake_diff[:,:2] = 0
                fake_diff[:,-2:] = 0
                IM = cv2.invertAffineTransform(M)
                img_white = np.full((aimg.shape[0],aimg.shape[1]), 255, dtype=np.float32)
                bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
                img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
                fake_diff = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
                img_white[img_white>20] = 255
                fthresh = 10
                fake_diff[fake_diff<fthresh] = 0
                fake_diff[fake_diff>=fthresh] = 255
                img_mask = img_white
                mask_h_inds, mask_w_inds = np.where(img_mask==255)
                mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
                mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
                mask_size = int(np.sqrt(mask_h*mask_w))
                k = max(mask_size//10, 10)
                #k = max(mask_size//20, 6)
                #k = 6
                kernel = np.ones((k,k),np.uint8)
                img_mask = cv2.erode(img_mask,kernel,iterations = 1)
                kernel = np.ones((2,2),np.uint8)
                fake_diff = cv2.dilate(fake_diff,kernel,iterations = 1)
                k = max(mask_size//20, 5)
                #k = 3
                #k = 3
                kernel_size = (k, k)
                blur_size = tuple(2*i+1 for i in kernel_size)
                img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
                k = 5
                kernel_size = (k, k)
                blur_size = tuple(2*i+1 for i in kernel_size)
                fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
                img_mask /= 255
                fake_diff /= 255
                #img_mask = fake_diff
                img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
                fake_merged = img_mask * bgr_fake + (1-img_mask) * target_img.astype(np.float32)
                fake_merged = fake_merged.astype(np.uint8)

                fake_images.append(fake_merged)
                
        if is_single:
            return fake_images[0]
        else:
            return fake_images[:return_count]

    #https://github.com/deepinsight/insightface/blob/4aa1a40b1f35633608e14e173d0f99f0f667321c/python-package/insightface/utils/face_align.py#L11
    def estimate_norm(self, lmk, image_size=112,mode='arcface'):
        assert lmk.shape == (5, 2)
        assert image_size%112==0 or image_size%128==0
        if image_size%112==0:
            ratio = float(image_size)/112.0
            diff_x = 0
        else:
            ratio = float(image_size)/128.0
            diff_x = 8.0*ratio
        dst = arcface_dst * ratio
        dst[:,0] += diff_x
        tform = trans.SimilarityTransform()
        tform.estimate(lmk, dst)
        M = tform.params[0:2, :]
        return M

    #https://github.com/deepinsight/insightface/blob/4aa1a40b1f35633608e14e173d0f99f0f667321c/python-package/insightface/utils/face_align.py#L32
    def norm_crop2(self, img, landmark, image_size=112, mode='arcface'):
        M = self.estimate_norm(landmark, image_size, mode)
        warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
        return warped, M

    def save_preds(self, preds, output_dir="output"):
        """
        Stacks all predicted images horizontally into one wide composite and saves to a single PNG.

        Args:
            preds: np.array of shape (batch_size, H, W, 3) in [0..1]
            output_dir: folder path where the PNG will be saved
        """
        os.makedirs(output_dir, exist_ok=True)


        preds = np.stack(preds, axis=0)

        batch_size = preds.shape[0]
        # Create a list of individual images from the batch
        columns = [preds[i] for i in range(batch_size)]
        
        # Concatenate them horizontally (axis=1)
        # Resulting shape => (H, batch_size * W, 3)
        composite_image = np.concatenate(columns, axis=1)

        # Scale to [0..255] and convert to uint8
        composite_image = np.clip(composite_image * 255, 0, 255).astype(np.uint8)

        # Create a random 6-character suffix for filename uniqueness
        random_suffix = uuid.uuid4().hex[:6]
        output_path = os.path.join(
            output_dir, f"inference_preds_{random_suffix}.png"
        )
        # Convert array -> PIL image -> save
        keras_image.array_to_img(composite_image).save(output_path)
        print(f"Predictions saved to: {output_path}")

    def compare_model_weights(self, model_before):
        changes = {}
        
        # Iterate through corresponding layers
        for layer_before, layer_after in zip(model_before.layers, self.trainable_model.generator.layers):
            weights_before = layer_before.get_weights()
            weights_after = layer_after.get_weights()
            
            if weights_before and weights_after:  # if layer has weights
                diff_norm = 0.0
                base_norm = 0.0
                # Compare each weight tensor in the layer
                for wb, wa in zip(weights_before, weights_after):
                    diff_norm += np.linalg.norm(wa - wb)
                    base_norm += np.linalg.norm(wb)
                    
                # Avoid division by zero; if base_norm is 0, use diff_norm directly.
                relative_change = diff_norm / base_norm if base_norm > 0 else diff_norm
                changes[layer_before.name] = relative_change
            else:
                # Layers without weights can be skipped or recorded as zero change.
                changes[layer_before.name] = 0.0
        
        return changes


    def tune_for(self, dataset, epochs=1, callbacks=[]):
        model_before = get_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), num_blocks=NUM_BLOCKS)
        model_before.set_weights(self.trainable_model.generator.get_weights())

        self.trainable_model.fit(
            dataset,
            #validation_data=validation,  # optional
            epochs=epochs,
            callbacks=callbacks
        )

        changes = self.compare_model_weights(model_before)

        # Optionally, sort and print layers by how much they changed
        sorted_changes = sorted(changes.items(), key=lambda item: item[1], reverse=True)
        for layer_name, change in sorted_changes:
            print(f"Layer: {layer_name}, Relative Change: {change:.4f}")