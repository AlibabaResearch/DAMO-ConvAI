import torch

if __name__ == "__main__":
    print(f"Cuda available: {torch.cuda.is_available()}")

    device_count = torch.cuda.device_count()
    print(f"Cuda device count: {device_count}")

    for idx in range(device_count):
        print(f"Device IDX: {idx}")
        print(f"Name: {torch.cuda.get_device_name(idx)}")
        print(f"Process: {torch.cuda.get_device_properties(idx)}", flush=True)
