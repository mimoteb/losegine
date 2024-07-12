import torch

def check_gpu():
    if torch.cuda.is_available():
        print("GPU is available.")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Current Device: {torch.cuda.current_device()}")
    else:
        print("GPU is not available.")

if __name__ == "__main__":
    check_gpu()
