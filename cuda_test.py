import torch

def test_cuda():
    if torch.cuda.is_available():
        print("✅ CUDA is available. GPU will be used.")
    else:
        print("❌ CUDA is not available. CPU will be used.")

if __name__ == "__main__":
    test_cuda()