import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

z1 = torch.empty(3)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(z1)
    z2 = torch.add(x, y, out=z1)
    print(z1)
