import pdb

import matplotlib
import numpy as np
from absl import app

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import reconstruction
import segmentation
from utils import img_utils, io_utils, recon_utils


def main(argv):
    mat_dict = io_utils.import_mat("tmp/subject.mat")

    data_dis_high = mat_dict["data_dis_highkey"].T
    data_dis_low = mat_dict["data_dis_lowkey"].T
    data_gas = np.conjugate(mat_dict["data_gas"])
    data_dis = recon_utils.flatten_data(np.conjugate(mat_dict["data_dis_tot"]).T)
    # image_gas = mat_data["Gas_Image"]
    # image_rbc = mat_data["RBC_Tot"]
    traj_dis_low = mat_dict["traj_dis_lowkey"]
    traj_gas = mat_dict["traj_gas"]
    traj_dis_high = mat_dict["traj_dis_highkey"]
    traj_dis = mat_dict["traj_dis_keyhole"]
    image_gas = reconstruction.reconstruct(
        data=data_gas, traj=traj_gas, kernel_sharpness=0.25, kernel_extent=0.25 * 9
    )
    image_dissolved = reconstruction.reconstruct(
        data=data_dis, traj=traj_dis, kernel_sharpness=0.14, kernel_extent=0.14 * 9
    )
    image_dissolved_high = reconstruction.reconstruct(
        data=data_dis_high,
        traj=traj_dis_high,
        kernel_sharpness=0.14,
        kernel_extent=0.14 * 9,
    )
    image_dissolved_low = reconstruction.reconstruct(
        data=data_dis_low,
        traj=traj_dis_low,
        kernel_sharpness=0.14,
        kernel_extent=0.14 * 9,
    )
    mask = segmentation.predict(np.abs(image_gas), erosion=3)
    image_rbc_high, _ = img_utils.dixon_decomposition(
        image_gas=image_gas,
        image_dissolved=image_dissolved_high,
        mask=mask,
        rbc_m_ratio=0.2369,
    )
    image_rbc_low, _ = img_utils.dixon_decomposition(
        image_gas=image_gas,
        image_dissolved=image_dissolved_low,
        mask=mask,
        rbc_m_ratio=0.21,
    )
    image_rbc, _ = img_utils.dixon_decomposition(
        image_gas=image_gas,
        image_dissolved=image_dissolved,
        mask=mask,
        rbc_m_ratio=0.224,
    )
    image_rbc_osc = img_utils.calculate_rbc_oscillation(
        image_rbc_high, image_rbc_low, image_rbc, mask
    )
    io_utils.export_nii(np.abs(image_dissolved_high), "tmp/dissolved_high.nii")
    io_utils.export_nii(np.abs(image_rbc_high), "tmp/rbc_high.nii")
    io_utils.export_nii(np.abs(image_rbc_low), "tmp/rbc_low.nii")
    io_utils.export_nii(np.abs(image_rbc), "tmp/rbc.nii")
    io_utils.export_nii(mask.astype(float), "tmp/mask.nii")
    io_utils.export_nii(image_rbc_osc * mask, "tmp/osc.nii")
    io_utils.export_nii(image_gas, "tmp/gas.nii")
    io_utils.export_nii(image_dissolved, "tmp/dissolved.nii")
    # plt.hist(100 * image_rbc_osc[mask > 0].flatten(), 50)
    plt.hist(image_rbc_osc[mask > 0].flatten(), 50)
    # plt.xlim(-10, 30)
    plt.show()
    pdb.set_trace()


if __name__ == "__main__":
    app.run(main)
