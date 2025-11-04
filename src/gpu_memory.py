import pynvml
import tensorflow as tf

def get_total_gpu_memory_in_mb():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming you're using GPU 0
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_memory = mem_info.total / (1024 * 1024)  # Convert bytes to megabytes
    pynvml.nvmlShutdown()
    return total_memory


def limit_gpu_memory(fraction=0.5):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            total_memory = get_total_gpu_memory_in_mb()
            memory_limit = total_memory * fraction
            for gpu in gpus:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
            print(f"Memory limit set to {memory_limit} MB per GPU")
        except RuntimeError as e:
            print(e)