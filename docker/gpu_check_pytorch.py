import torch


def main():
    print("=" * 80)
    print(f"[pytorch check] pytorch version: >> {torch.__version__}")
    cuda_is_avail = torch.cuda.is_available()
    print(f"[pytorch check] CUDA is available {cuda_is_avail}")

    device_name = torch.cuda.get_device_name(0)
    print(f"[pytorch check] CUDA device found: {device_name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
