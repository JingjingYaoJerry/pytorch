import torch

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')

if __name__ == '__main__':
    print(f'Using device: {device}')