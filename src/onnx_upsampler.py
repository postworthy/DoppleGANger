import numpy as np
import tensorflow as tf
import onnxruntime as ort
import cv2
import grpc
import upscale_pb2
import upscale_pb2_grpc
import tfrecord_service_pb2
import tfrecord_service_pb2_grpc
from sr_model import FaceSuperResolutionModel

# Create the ONNX session (only once).
# onnx_session = ort.InferenceSession(
#     "GFPGANv1.4.onnx",
#     providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
# )
# input_name = onnx_session.get_inputs()[0].name

# def run_onnx(array_tensor):
#     """
#     This function is called via tf.py_function.
#     'array_tensor' is a tf.Tensor of shape [1,3,512,512].
#     We convert it to a NumPy array and pass it into onnx_session.run.
#     """
#     np_array = array_tensor.numpy()  # Convert TF tensor -> NumPy
#     # Now we have a proper NumPy array of shape [1,3,512,512].
#     result = onnx_session.run(None, {input_name: np_array})[0]
#     return result

# @tf.function(jit_compile=False)
# def onnx_upsample_oldest(y_pred):
#     """
#     y_pred: A batch of images, shape [B,256,256,3] in NHWC.
#     We'll:
#       1) For each image, upscale it to 512x512 (Lanczos3) in NHWC.
#       2) Transpose to NCHW + batch => [1,3,512,512].
#       3) Pass that to ONNX.
#       4) Convert the output back to NHWC and size 256x256.
#     """

#     def process_image(image):
#         # image: shape [256,256,3] for a single image in NHWC.

#         # 1) Upscale to 512x512
#         image_up = tf.image.resize(image, [512, 512], method=tf.image.ResizeMethod.LANCZOS3)
#         # 2) NHWC -> NCHW
#         image_nchw = tf.transpose(image_up, perm=[2, 0, 1])  # [3,512,512]
#         # 3) Add batch dim => [1,3,512,512]
#         image_input = tf.expand_dims(image_nchw, axis=0)

#         # 4) Call the ONNX model via tf.py_function
#         out = tf.py_function(
#             func=run_onnx,            # The function above
#             inp=[image_input],        # We pass just one tensor in a list
#             Tout=tf.float32
#         )
#         # onnxruntime returns shape [1,3,512,512].
#         out.set_shape([1, 3, 512, 512])

#         # 5) Squeeze batch => [3,512,512]
#         out = tf.squeeze(out, axis=0)

#         # 6) Convert back to NHWC => [512,512,3]
#         out_nhwc = tf.transpose(out, perm=[1, 2, 0])

#         # 7) Downscale to 256×256
#         final = tf.image.resize(out_nhwc, [256, 256], method=tf.image.ResizeMethod.LANCZOS3)
#         return final

#     # tf.map_fn processes the batch dimension of y_pred (shape [B,256,256,3]).
#     return tf.map_fn(process_image, y_pred, fn_output_signature=tf.float32)

# @tf.function(jit_compile=False)
# def onnx_upsample_old(y_pred):
#     # y_pred shape: [B, 256, 256, 3]

#     # 1) Upscale all images to 512x512
#     y_up = tf.image.resize(y_pred, [512, 512], method=tf.image.ResizeMethod.LANCZOS3)
    
#     # 2) Convert from NHWC to NCHW for the entire batch: [B, 512, 512, 3] -> [B, 3, 512, 512]
#     y_nchw = tf.transpose(y_up, perm=[0, 3, 1, 2])
    
#     # 3) Define a function to run the ONNX model on the full batch
#     def run_onnx_batch(array_tensor):
#     # array_tensor shape: [B, 3, 512, 512]
#         np_array = array_tensor.numpy()  # Convert to NumPy array
#         B = np_array.shape[0]
#         sub_batch_size = 1  # or another small number that your memory can handle
#         outputs = []onnx_upsample
#     # 5) Convert output from NCHW back to NHWC: [B, 3, 512, 512] -> [B, 512, 512, 3]
#     out_nhwc = tf.transpose(out_batch, perm=[0, 2, 3, 1])
    
#     # 6) Downscale all images back to 256x256
#     final = tf.image.resize(out_nhwc, [256, 256], method=tf.image.ResizeMethod.LANCZOS3)
    
#     return final

class Upsampler():
    
    def __init__(self):
        super(Upsampler, self).__init__()
        self.sr_model = FaceSuperResolutionModel()
        self.sr_model.load_weights("./models/sr_model/sr_model_final")
        print("Loaded SR Model: ./models/sr_model/sr_model_final")
        
    @tf.function(jit_compile=False)
    def upsample(self, y_pred, downscale=True, training=False):
        y_pred = tf.image.resize(y_pred, [256, 256], method="bicubic")
        sr = self.sr_model(y_pred, training=training)
        
        if downscale:
            out = tf.image.resize(sr, [256, 256], method="bicubic")
        else:
            out = tf.image.resize(sr, [512, 512], method="bicubic")

        # Set static shape information for TensorFlow.
        if downscale:
            out.set_shape([y_pred.shape[0], 256, 256, 3])
        else:
            out.set_shape([y_pred.shape[0], 512, 512, 3])
        
        # Optionally, convert output back to float in the range [0, 1].
        #out_float = tf.cast(out, tf.float32) / 255.0
        out_float = tf.cast(out, tf.float32)
        return out_float


#######
#
# You need to have the FaceUpscalerService running for this remote upsampler to work
#
#######
class Upsampler_Remote():    

    @tf.function(jit_compile=False)
    def upsample(self, y_pred, downscale=True):
        def call_grpc_upscale(np_y_pred):
            # np_y_pred shape: [B, 256, 256, 3] with float32 values in [0, 1]
            # Convert to uint8 images in the 0–255 range for OpenCV operations.
            np_y_pred_uint8 = np.clip(np_y_pred * 255.0, 0, 255).astype(np.uint8)
            
            # Prepare a list to hold JPEG-encoded images.
            images_bytes = []
            for i in range(np_y_pred_uint8.shape[0]):
                image = np_y_pred_uint8[i]  # shape: (256, 256, 3)
                # Upscale the image to 512x512 using Lanczos interpolation.
                image_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
                # Encode the image as JPEG.
                ret, buf = cv2.imencode('.jpg', image_resized)
                if not ret:
                    raise ValueError("Failed to encode the image to JPEG.")
                images_bytes.append(buf.tobytes())
            
            # Create a channel with increased message size limits.
            channel = grpc.insecure_channel(
                'localhost:50051',
                options=[
                    ('grpc.max_receive_message_length', 10 * 1024 * 1024),
                    ('grpc.max_send_message_length', 10 * 1024 * 1024)
                ]
            )
            stub = upscale_pb2_grpc.UpscaleServiceStub(channel)
            
            # Construct the request.
            request = upscale_pb2.UpscaleRequest(
                has_aligned=False,
                upscale_factor=2,
                images=images_bytes
            )
            # Send the request to the service.
            response = stub.UpscaleImages(request)
            
            # Process the response:
            processed_images = []
            for img_bytes in response.processed_images:
                # Decode the JPEG back to an image.
                np_arr = np.frombuffer(img_bytes, np.uint8)
                decoded_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if decoded_img is None:
                    raise ValueError("Failed to decode the processed image.")
                processed_images.append(decoded_img)
            
            # Optionally, downscale back to 256x256 to mimic your previous behavior.
            if downscale:
                processed_images_downscaled = [
                    cv2.resize(img, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                    for img in processed_images
                ]
                
                # Stack the list of images into a NumPy array of shape [B, 256, 256, 3].
                return np.stack(processed_images_downscaled, axis=0)
            else:
                processed_images_downscaled = [
                    cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
                    for img in processed_images
                ]

                return np.stack(processed_images_downscaled, axis=0)
        
        y_pred = tf.image.resize(y_pred, [256, 256], method="bicubic")
        # Use tf.py_function to wrap the non-TensorFlow code.
        out = tf.py_function(func=call_grpc_upscale, inp=[y_pred], Tout=tf.uint8)
        
        # Set static shape information for TensorFlow.
        if downscale:
            out.set_shape([y_pred.shape[0], 256, 256, 3])
        else:
            out.set_shape([y_pred.shape[0], 512, 512, 3])
        
        # Optionally, convert output back to float in the range [0, 1].
        out_float = tf.cast(out, tf.float32) / 255.0
        return out_float

    def initialize_upsampler(self, tfrecord_name):
        # Create a channel to the server.
        channel = grpc.insecure_channel('localhost:50051')
        stub = tfrecord_service_pb2_grpc.TFRecordServiceStub(channel)
        
        # Build and send the LoadTFRecord request.
        request = tfrecord_service_pb2.LoadTFRecordRequest(tfrecord_name=tfrecord_name)
        response = stub.LoadTFRecord(request)
        #print("LoadTFRecord response:", response.message)
        # Close the channel or reuse it as needed.
        channel.close()

    def get_upsampled_record(self, tfrecord_name: str, record_index: int) -> np.ndarray:
        """
        Fetches a single upscaled record from the TFRecord service and returns it
        as a NumPy array of shape [H, W, 3], dtype float32 in [0,1].
        
        Args:
            tfrecord_name: name of the TFRecord file previously loaded on the server.
            record_index: zero-based index of the record to retrieve.
        
        Returns:
            A NumPy array of the image, normalized to [0.0,1.0].
        
        Raises:
            RuntimeError on gRPC errors or if the server returns no image.
        """
        # 1) Open gRPC channel
        channel = grpc.insecure_channel('localhost:50051')
        stub = tfrecord_service_pb2_grpc.TFRecordServiceStub(channel)
        
        # 2) Build and send the request
        request = tfrecord_service_pb2.GetRecordRequest(
            tfrecord_name=tfrecord_name,
            record_index=record_index
        )
        response = stub.GetRecord(request)
        channel.close()
        
        # 3) Ensure we got bytes back
        if not response.processed_images:
            raise RuntimeError(f"No image returned for record {record_index}")
        
        imgs = []
        
        for jpg in response.processed_images:
            arr = np.frombuffer(jpg, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            imgs.append(img.astype(np.float32)/255.0)  # shape (256,256,3)
        
        batch_array = np.stack(imgs, axis=0)

        return batch_array
