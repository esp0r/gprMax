#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

namespace py = pybind11;

void update_electric(
    int nx,
    int ny,
    int nz,
    int nthreads,
    py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsE,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ID,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ex,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ey,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ez,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hx,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hy,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hz
) {
    auto updatecoeffsE_ = updatecoeffsE.unchecked<2>();
    auto ID_ = ID.unchecked<4>();
    auto Ex_ = Ex.mutable_unchecked<3>();
    auto Ey_ = Ey.mutable_unchecked<3>();
    auto Ez_ = Ez.mutable_unchecked<3>();
    auto Hx_ = Hx.unchecked<3>();
    auto Hy_ = Hy.unchecked<3>();
    auto Hz_ = Hz.unchecked<3>();

    int materialEx, materialEy, materialEz;

    omp_set_num_threads(nthreads);

    // 2D - Ex component
    if (nx == 1) {
        #pragma omp parallel for schedule(static) private(materialEx)
        for (int j = 1; j < ny; ++j) {
            for (int k = 1; k < nz; ++k) {
                materialEx = ID_(0, 0, j, k);
                Ex_(0, j, k) = (updatecoeffsE_(materialEx, 0) * Ex_(0, j, k) +
                                updatecoeffsE_(materialEx, 2) * (Hz_(0, j, k) - Hz_(0, j - 1, k)) -
                                updatecoeffsE_(materialEx, 3) * (Hy_(0, j, k) - Hy_(0, j, k - 1)));
            }
        }
    }

    // 2D - Ey component
    else if (ny == 1) {
        #pragma omp parallel for schedule(static) private(materialEy)
        for (int i = 1; i < nx; ++i) {
            for (int k = 1; k < nz; ++k) {
                materialEy = ID_(1, i, 0, k);
                Ey_(i, 0, k) = (updatecoeffsE_(materialEy, 0) * Ey_(i, 0, k) +
                                updatecoeffsE_(materialEy, 3) * (Hx_(i, 0, k) - Hx_(i, 0, k - 1)) -
                                updatecoeffsE_(materialEy, 1) * (Hz_(i, 0, k) - Hz_(i - 1, 0, k)));
            }
        }
    }

    // 2D - Ez component
    else if (nz == 1) {
        #pragma omp parallel for schedule(static) private(materialEz)
        for (int i = 1; i < nx; ++i) {
            for (int j = 1; j < ny; ++j) {
                materialEz = ID_(2, i, j, 0);
                Ez_(i, j, 0) = (updatecoeffsE_(materialEz, 0) * Ez_(i, j, 0) +
                                updatecoeffsE_(materialEz, 1) * (Hy_(i, j, 0) - Hy_(i - 1, j, 0)) -
                                updatecoeffsE_(materialEz, 2) * (Hx_(i, j, 0) - Hx_(i, j - 1, 0)));
            }
        }
    }

    // 3D
    else {
        #pragma omp parallel for schedule(static) private(materialEx, materialEy, materialEz)
        for (int i = 1; i < nx; ++i) {
            for (int j = 1; j < ny; ++j) {
                for (int k = 1; k < nz; ++k) {
                    materialEx = ID_(0, i, j, k);
                    materialEy = ID_(1, i, j, k);
                    materialEz = ID_(2, i, j, k);
                    Ex_(i, j, k) = (updatecoeffsE_(materialEx, 0) * Ex_(i, j, k) +
                                    updatecoeffsE_(materialEx, 2) * (Hz_(i, j, k) - Hz_(i, j - 1, k)) -
                                    updatecoeffsE_(materialEx, 3) * (Hy_(i, j, k) - Hy_(i, j, k - 1)));
                    Ey_(i, j, k) = (updatecoeffsE_(materialEy, 0) * Ey_(i, j, k) +
                                    updatecoeffsE_(materialEy, 3) * (Hx_(i, j, k) - Hx_(i, j, k - 1)) -
                                    updatecoeffsE_(materialEy, 1) * (Hz_(i, j, k) - Hz_(i - 1, j, k)));
                    Ez_(i, j, k) = (updatecoeffsE_(materialEz, 0) * Ez_(i, j, k) +
                                    updatecoeffsE_(materialEz, 1) * (Hy_(i, j, k) - Hy_(i - 1, j, k)) -
                                    updatecoeffsE_(materialEz, 2) * (Hx_(i, j, k) - Hx_(i, j - 1, k)));
                }
            }
        }

        // Ex components at i = 0
        #pragma omp parallel for schedule(static) private(materialEx)
        for (int j = 1; j < ny; ++j) {
            for (int k = 1; k < nz; ++k) {
                materialEx = ID_(0, 0, j, k);
                Ex_(0, j, k) = (updatecoeffsE_(materialEx, 0) * Ex_(0, j, k) +
                                updatecoeffsE_(materialEx, 2) * (Hz_(0, j, k) - Hz_(0, j - 1, k)) -
                                updatecoeffsE_(materialEx, 3) * (Hy_(0, j, k) - Hy_(0, j, k - 1)));
            }
        }

        // Ey components at j = 0
        #pragma omp parallel for schedule(static) private(materialEy)
        for (int i = 1; i < nx; ++i) {
            for (int k = 1; k < nz; ++k) {
                materialEy = ID_(1, i, 0, k);
                Ey_(i, 0, k) = (updatecoeffsE_(materialEy, 0) * Ey_(i, 0, k) +
                                updatecoeffsE_(materialEy, 3) * (Hx_(i, 0, k) - Hx_(i, 0, k - 1)) -
                                updatecoeffsE_(materialEy, 1) * (Hz_(i, 0, k) - Hz_(i - 1, 0, k)));
            }
        }

        // Ez components at k = 0
        #pragma omp parallel for schedule(static) private(materialEz)
        for (int i = 1; i < nx; ++i) {
            for (int j = 1; j < ny; ++j) {
                materialEz = ID_(2, i, j, 0);
                Ez_(i, j, 0) = (updatecoeffsE_(materialEz, 0) * Ez_(i, j, 0) +
                                updatecoeffsE_(materialEz, 1) * (Hy_(i, j, 0) - Hy_(i - 1, j, 0)) -
                                updatecoeffsE_(materialEz, 2) * (Hx_(i, j, 0) - Hx_(i, j - 1, 0)));
            }
        }
    }
}

void update_magnetic(
    int nx,
    int ny,
    int nz,
    int nthreads,
    py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsH,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ID,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ex,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ey,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ez,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hx,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hy,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hz
) {
    auto updatecoeffsH_ = updatecoeffsH.unchecked<2>();
    auto ID_ = ID.unchecked<4>();
    auto Ex_ = Ex.unchecked<3>();
    auto Ey_ = Ey.unchecked<3>();
    auto Ez_ = Ez.unchecked<3>();
    auto Hx_ = Hx.mutable_unchecked<3>();
    auto Hy_ = Hy.mutable_unchecked<3>();
    auto Hz_ = Hz.mutable_unchecked<3>();

    int materialHx, materialHy, materialHz;

    omp_set_num_threads(nthreads);

    if (nx == 1 || ny == 1 || nz == 1) {
        // 2D case
        if (ny == 1 || nz == 1) {
            #pragma omp parallel for schedule(static) private(materialHx)
            for (int i = 1; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    for (int k = 0; k < nz; ++k) {
                        materialHx = ID_(3, i, j, k);
                        Hx_(i, j, k) = (updatecoeffsH_(materialHx, 0) * Hx_(i, j, k) -
                                        updatecoeffsH_(materialHx, 2) * (Ez_(i, j + 1, k) - Ez_(i, j, k)) +
                                        updatecoeffsH_(materialHx, 3) * (Ey_(i, j, k + 1) - Ey_(i, j, k)));
                    }
                }
            }
        }

        if (nx == 1 || nz == 1) {
            #pragma omp parallel for schedule(static) private(materialHy)
            for (int i = 0; i < nx; ++i) {
                for (int j = 1; j < ny; ++j) {
                    for (int k = 0; k < nz; ++k) {
                        materialHy = ID_(4, i, j, k);
                        Hy_(i, j, k) = (updatecoeffsH_(materialHy, 0) * Hy_(i, j, k) -
                                        updatecoeffsH_(materialHy, 3) * (Ex_(i, j, k + 1) - Ex_(i, j, k)) +
                                        updatecoeffsH_(materialHy, 1) * (Ez_(i + 1, j, k) - Ez_(i, j, k)));
                    }
                }
            }
        }

        if (nx == 1 || ny == 1) {
            #pragma omp parallel for schedule(static) private(materialHz)
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    for (int k = 1; k < nz; ++k) {
                        materialHz = ID_(5, i, j, k);
                        Hz_(i, j, k) = (updatecoeffsH_(materialHz, 0) * Hz_(i, j, k) -
                                        updatecoeffsH_(materialHz, 1) * (Ey_(i + 1, j, k) - Ey_(i, j, k)) +
                                        updatecoeffsH_(materialHz, 2) * (Ex_(i, j + 1, k) - Ex_(i, j, k)));
                    }
                }
            }
        }
    } else {
        // 3D case
        #pragma omp parallel for schedule(static) private(materialHx, materialHy, materialHz)
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < nz; ++k) {
                    materialHx = ID_(3, i + 1, j, k);
                    materialHy = ID_(4, i, j + 1, k);
                    materialHz = ID_(5, i, j, k + 1);
                    Hx_(i + 1, j, k) = (updatecoeffsH_(materialHx, 0) * Hx_(i + 1, j, k) -
                                        updatecoeffsH_(materialHx, 2) * (Ez_(i + 1, j + 1, k) - Ez_(i + 1, j, k)) +
                                        updatecoeffsH_(materialHx, 3) * (Ey_(i + 1, j, k + 1) - Ey_(i + 1, j, k)));
                    Hy_(i, j + 1, k) = (updatecoeffsH_(materialHy, 0) * Hy_(i, j + 1, k) -
                                        updatecoeffsH_(materialHy, 3) * (Ex_(i, j + 1, k + 1) - Ex_(i, j + 1, k)) +
                                        updatecoeffsH_(materialHy, 1) * (Ez_(i + 1, j + 1, k) - Ez_(i, j + 1, k)));
                    Hz_(i, j, k + 1) = (updatecoeffsH_(materialHz, 0) * Hz_(i, j, k + 1) -
                                        updatecoeffsH_(materialHz, 1) * (Ey_(i + 1, j, k + 1) - Ey_(i, j, k + 1)) +
                                        updatecoeffsH_(materialHz, 2) * (Ex_(i, j + 1, k + 1) - Ex_(i, j, k + 1)));
                }
            }
        }
    }
}

PYBIND11_MODULE(pybind11_fields_updates_normal, m) {
    m.def("update_electric", &update_electric, py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("nthreads"),
          py::arg("updatecoeffsE"), py::arg("ID"), py::arg("Ex"), py::arg("Ey"), py::arg("Ez"),
          py::arg("Hx"), py::arg("Hy"), py::arg("Hz"));
    m.def("update_magnetic", &update_magnetic, py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("nthreads"),
          py::arg("updatecoeffsH"), py::arg("ID"), py::arg("Ex"), py::arg("Ey"), py::arg("Ez"),
          py::arg("Hx"), py::arg("Hy"), py::arg("Hz"));
}
