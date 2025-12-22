import torch

# Быстрый просмотр
mean = torch.load("datasets/rocstories/statistics-old/encodings-bert-base-cased-mean.pt", map_location=torch.device('cpu'))
std = torch.load("datasets/rocstories/statistics-old/encodings-bert-base-cased-std.pt", map_location=torch.device('cpu'))

print(f"Mean shape: {mean.shape}, values: {mean[:5]}")
print(f"Std shape: {std.shape}, values: {std[:5]}")

print('NEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEW')

# Быстрый просмотр
mean = torch.load("datasets/rocstories/statistics/encodings-bert-base-cased-mean.pt", map_location=torch.device('cpu'))
std = torch.load("datasets/rocstories/statistics/encodings-bert-base-cased-std.pt", map_location=torch.device('cpu'))

print(f"Mean shape: {mean.shape}, values: {mean[:5]}")
print(f"Std shape: {std.shape}, values: {std[:5]}")