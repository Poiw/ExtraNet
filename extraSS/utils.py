import imageio

def tensorToNumpy(x):
    x = x.detach().cpu().numpy()
    x = x.transpose([1, 2, 0])

    return x

def save_exr(img, path):
    imageio.imwrite(path, img)