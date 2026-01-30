import torch
import numpy as np
from scipy import linalg

class Compute_lip_metric():
    """
    Input to the loss must be in mm
    """
    def __init__(self, vertice_dim, 
                        loss_creterion,
                        loss_dict={}):
        self.vertice_dim = vertice_dim
        self.loss = loss_creterion
        self.loss_dict = loss_dict

        self.lip_vertices = {"upper": [3506], "lower": [3531]}
        self.lip_traiangle = {"upper": [2866, 3546, 1751, 3531],
                              "lower": [2938, 3504, 1849, 3506]}

        fd_config = loss_dict.get("fd_config", {})
        self.hist_max = fd_config.get("hist_max", 50)
        self.hist_num_bins = fd_config.get("hist_num_bins", 100)

    def get_distance(self, diff):
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        return dist

    def lip_differences_using_triangle(self, predict, real, in_mm=True):
        """
        predict: T x vert * 3
        real: T x vert * 3
        """

        scale = 1000.0 if in_mm else 1.0

        pred_verts_mm = predict.view(-1, self.vertice_dim // 3, 3) * scale
        gt_verts_mm = real.view(-1, self.vertice_dim // 3, 3) * scale

        # extract the upper and lower lip
        upper_lip = pred_verts_mm[:, self.lip_traiangle["upper"]]
        lower_lip = pred_verts_mm[:, self.lip_traiangle["lower"]]
        pred_distance_in_mm = self.get_distance(upper_lip-lower_lip) # Nf x 4
        pred_distance_in_mm = torch.mean(pred_distance_in_mm, dim=-1) # Nf x 1

        upper_lip = gt_verts_mm[:, self.lip_traiangle["upper"]]
        lower_lip = gt_verts_mm[:, self.lip_traiangle["lower"]]
        gt_distance_in_mm = self.get_distance(upper_lip-lower_lip) # Nf x 4
        gt_distance_in_mm = torch.mean(gt_distance_in_mm, dim=-1) # Nf x 1

        return gt_distance_in_mm, pred_distance_in_mm

    def lip_differences_using_single_point_rep(self, predict, real):
        """
        predict: T x vert * 3
        real: T x vert * 3
        """
        pred_verts_mm = predict.view(-1, self.vertice_dim // 3, 3) * 1000.0
        gt_verts_mm = real.view(-1, self.vertice_dim // 3, 3) * 1000.0

        # extract the upper and lower lip
        upper_lip = pred_verts_mm[:, self.lip_vertices["upper"]]
        lower_lip = pred_verts_mm[:, self.lip_vertices["lower"]]
        pred_distance_in_mm = self.get_distance(upper_lip-lower_lip) # Nf x 1

        upper_lip = gt_verts_mm[:, self.lip_vertices["upper"]]
        lower_lip = gt_verts_mm[:, self.lip_vertices["lower"]]
        gt_distance_in_mm = self.get_distance(upper_lip-lower_lip) # Nf x 1

        return gt_distance_in_mm, pred_distance_in_mm

    def dist_to_bins(self, dist, num_bin=10, min=0.0, max=10.0):
        """
        Given the lip differences as distanc, quantize them into bins
        """
        bins = torch.histc(dist, num_bin)
        # bins = torch.histogram(dist, num_bin)
        return bins

    def plot_bins(self, pred_bin, gt_bin):
        """
        Given the bins of the prediction and real, plot on the graph and store ir
        """
        pass

    def compute_FD_distance(self, pred_bin, gt_bin):
        """
        Given the bins of the pred and real, compute the FD score between them
        """
        if torch.is_tensor(pred_bin):
            pred_bin=pred_bin.cpu().numpy()
            gt_bin=gt_bin.cpu().numpy()

        # calculate mean and covariance statistics
        mu1, sigma1 = gt_bin.mean(axis=0), np.cov(gt_bin, rowvar=False)
        mu2, sigma2 = pred_bin.mean(axis=0), np.cov(pred_bin,  rowvar=False)

        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        import pdb; pdb.set_trace()
        covmean = linalg.sqrtm(sigma1.dot(sigma2))

        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def compute_FD_distance2(self, pred_bin, gt_bin, eps=1e-12):
        """
        Given the bins of the pred and real, compute the FD score between them
        """
        if torch.is_tensor(pred_bin):
            pred_bin=pred_bin.detach().cpu().numpy()
            gt_bin=gt_bin.detach().cpu().numpy()


        # calculate mean and covariance statistics
        mu1, sigma1 = gt_bin.mean(axis=0), np.cov(gt_bin, rowvar=False)
        mu2, sigma2 = pred_bin.mean(axis=0), np.cov(pred_bin,  rowvar=False)

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

        diff = mu1 - mu2
        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def compute_FD_from_pred_real(self, pred, real):

        # compute the lip difference
        pred_diff_dist, gt_diff_dist = self.lip_differences_using_triangle(pred, real)
        # convert to bins
        pred_diff_bins = torch.histc(pred_diff_dist, self.hist_num_bins, min=0, max=self.hist_max) / pred_diff_dist.shape[0]
        gt_diff_bins = torch.histc(gt_diff_dist, self.hist_num_bins, min=0, max=self.hist_max) / gt_diff_dist.shape[0]
        # compute the FD score
        FD_score = self.compute_FD_distance2(pred_diff_bins, gt_diff_bins)
        return FD_score