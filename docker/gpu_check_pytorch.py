import torch


def main():
    print(f"pytorch version: {torch.__version__}")
    cuda_is_avail = torch.cuda.is_available()
    print(f"CUDA is available {cuda_is_avail}")

    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA device found: {device_name}")


if __name__ == "__main__":
    main()
