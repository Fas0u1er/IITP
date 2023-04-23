from geomloss import SamplesLoss
from utils import img_to_weighted_sample


def sinkhorn(img1, img2):
    s1, w1 = img_to_weighted_sample(img1)
    s2, w2 = img_to_weighted_sample(img2)

    loss = SamplesLoss("sinkhorn", p=1, blur=0.005, backend="multiscale", scaling=0.5, debias=False)

    return loss(w1, s1, w2, s2).item()
