import pandas as pd
import matplotlib.pyplot as plt


class pixel:
    def __init__(self, coord_id, d_cov, d_met, d_yie):
        self.coord_id = coord_id
        self.cov = d_cov[d_cov.coord_id == coord_id]
        self.yie = d_yie[d_yie.coord_id == coord_id]
        self.FID = self.cov.FID  # can take instead: set(...)
        self.met = d_met[d_met.FID.isin(set(self.FID))]

    # printing method:
    def __str__(self):
        return "FID:  " + str(set(self.FID)) + "--------------------------" + "\n" + "yield: " + str(self.yie) + "\n" + "coord_id: " + self.coord_id + "\n"

    def __repr__(self):
        return self.__str__()

    def ndvi(self):
        # NDVI := NIR(Band8)-Red(Band4)/NIR(Band8)+Red(Band4)
        self.ndvi = (self.cov.B08 - self.cov.B04) / \
            (self.cov.B08 + self.cov.B04)
        return self.ndvi

    def plot_ndvi(self):
        try:
            plt.scatter(self.cov.date, self.ndvi)
        except:
            self.ndvi()
            plt.scatter(self.cov.date, self.ndvi())
        plt.ylabel("NDVI")
        plt.ylim([0, 1])
        plt.xlabel("date")
        plt.show()


def random_pixel(d_cov, d_met, d_yie, n=1):
    result = []
    cid = d_cov.coord_id.to_frame().sample(n, ignore_index=True).coord_id
    for i in range(n):
        result.append(pixel(cid[i], d_cov, d_met, d_yie))
    return result
