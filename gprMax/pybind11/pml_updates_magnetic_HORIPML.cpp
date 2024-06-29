#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

namespace py = pybind11;

void order1_xminus(
    int xs,
    int xf,
    int ys,
    int yf,
    int zs,
    int zf,
    int nthreads,
    py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsH,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ID,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ex,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ey,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ez,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hx,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hy,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hz,
    py::array_t<float, py::array::c_style | py::array::forcecast> Phi1,
    py::array_t<float, py::array::c_style | py::array::forcecast> Phi2,
    py::array_t<float, py::array::c_style | py::array::forcecast> RA,
    py::array_t<float, py::array::c_style | py::array::forcecast> RB,
    py::array_t<float, py::array::c_style | py::array::forcecast> RE,
    py::array_t<float, py::array::c_style | py::array::forcecast> RF,
    float d
) {
    auto updatecoeffsH_ = updatecoeffsH.unchecked<2>();
    auto ID_ = ID.unchecked<4>();
    auto Ex_ = Ex.mutable_unchecked<3>();
    auto Ey_ = Ey.mutable_unchecked<3>();
    auto Ez_ = Ez.mutable_unchecked<3>();
    auto Hx_ = Hx.unchecked<3>();
    auto Hy_ = Hy.mutable_unchecked<3>();
    auto Hz_ = Hz.mutable_unchecked<3>();
    auto Phi1_ = Phi1.mutable_unchecked<4>();
    auto Phi2_ = Phi2.mutable_unchecked<4>();
    auto RA_ = RA.unchecked<2>();
    auto RB_ = RB.unchecked<2>();
    auto RE_ = RE.unchecked<2>();
    auto RF_ = RF.unchecked<2>();

    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    omp_set_num_threads(nthreads);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nx; ++i) {
        int ii = xf - (i + 1);
        float RA01 = RA_(0, i) - 1;
        float RB0 = RB_(0, i);
        float RE0 = RE_(0, i);
        float RF0 = RF_(0, i);

        for (int j = 0; j < ny; ++j) {
            int jj = j + ys;
            for (int k = 0; k < nz; ++k) {
                int kk = k + zs;

                // Hy
                int materialHy = ID_(4, ii, jj, kk);
                float dEz = (Ez_(ii + 1, jj, kk) - Ez_(ii, jj, kk)) / d;
                Hy_(ii, jj, kk) = (Hy_(ii, jj, kk) + updatecoeffsH_(materialHy, 4) *
                                  (RA01 * dEz + RB0 * Phi1_(0, i, j, k)));
                Phi1_(0, i, j, k) = RE0 * Phi1_(0, i, j, k) - RF0 * dEz;

                // Hz
                int materialHz = ID_(5, ii, jj, kk);
                float dEy = (Ey_(ii + 1, jj, kk) - Ey_(ii, jj, kk)) / d;
                Hz_(ii, jj, kk) = (Hz_(ii, jj, kk) - updatecoeffsH_(materialHz, 4) *
                                  (RA01 * dEy + RB0 * Phi2_(0, i, j, k)));
                Phi2_(0, i, j, k) = RE0 * Phi2_(0, i, j, k) - RF0 * dEy;
            }
        }
    }
}

void order1_xplus(
    int xs,
    int xf,
    int ys,
    int yf,
    int zs,
    int zf,
    int nthreads,
    py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsH,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ID,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ex,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ey,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ez,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hx,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hy,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hz,
    py::array_t<float, py::array::c_style | py::array::forcecast> Phi1,
    py::array_t<float, py::array::c_style | py::array::forcecast> Phi2,
    py::array_t<float, py::array::c_style | py::array::forcecast> RA,
    py::array_t<float, py::array::c_style | py::array::forcecast> RB,
    py::array_t<float, py::array::c_style | py::array::forcecast> RE,
    py::array_t<float, py::array::c_style | py::array::forcecast> RF,
    float d
) {
    auto updatecoeffsH_ = updatecoeffsH.unchecked<2>();
    auto ID_ = ID.unchecked<4>();
    auto Ex_ = Ex.mutable_unchecked<3>();
    auto Ey_ = Ey.mutable_unchecked<3>();
    auto Ez_ = Ez.mutable_unchecked<3>();
    auto Hx_ = Hx.unchecked<3>();
    auto Hy_ = Hy.mutable_unchecked<3>();
    auto Hz_ = Hz.mutable_unchecked<3>();
    auto Phi1_ = Phi1.mutable_unchecked<4>();
    auto Phi2_ = Phi2.mutable_unchecked<4>();
    auto RA_ = RA.unchecked<2>();
    auto RB_ = RB.unchecked<2>();
    auto RE_ = RE.unchecked<2>();
    auto RF_ = RF.unchecked<2>();

    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    omp_set_num_threads(nthreads);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nx; ++i) {
        int ii = i + xs;
        float RA01 = RA_(0, i) - 1;
        float RB0 = RB_(0, i);
        float RE0 = RE_(0, i);
        float RF0 = RF_(0, i);

        for (int j = 0; j < ny; ++j) {
            int jj = j + ys;
            for (int k = 0; k < nz; ++k) {
                int kk = k + zs;

                // Hy
                int materialHy = ID_(4, ii, jj, kk);
                float dEz = (Ez_(ii + 1, jj, kk) - Ez_(ii, jj, kk)) / d;
                Hy_(ii, jj, kk) += updatecoeffsH_(materialHy, 4) * (RA01 * dEz + RB0 * Phi1_(0, i, j, k));
                Phi1_(0, i, j, k) = RE0 * Phi1_(0, i, j, k) - RF0 * dEz;

                // Hz
                int materialHz = ID_(5, ii, jj, kk);
                float dEy = (Ey_(ii + 1, jj, kk) - Ey_(ii, jj, kk)) / d;
                Hz_(ii, jj, kk) -= updatecoeffsH_(materialHz, 4) * (RA01 * dEy + RB0 * Phi2_(0, i, j, k));
                Phi2_(0, i, j, k) = RE0 * Phi2_(0, i, j, k) - RF0 * dEy;
            }
        }
    }
}

void order1_yminus(
    int xs,
    int xf,
    int ys,
    int yf,
    int zs,
    int zf,
    int nthreads,
    py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsH,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ID,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ex,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ey,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ez,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hx,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hy,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hz,
    py::array_t<float, py::array::c_style | py::array::forcecast> Phi1,
    py::array_t<float, py::array::c_style | py::array::forcecast> Phi2,
    py::array_t<float, py::array::c_style | py::array::forcecast> RA,
    py::array_t<float, py::array::c_style | py::array::forcecast> RB,
    py::array_t<float, py::array::c_style | py::array::forcecast> RE,
    py::array_t<float, py::array::c_style | py::array::forcecast> RF,
    float d
) {
    auto updatecoeffsH_ = updatecoeffsH.unchecked<2>();
    auto ID_ = ID.unchecked<4>();
    auto Ex_ = Ex.mutable_unchecked<3>();
    auto Ey_ = Ey.mutable_unchecked<3>();
    auto Ez_ = Ez.mutable_unchecked<3>();
    auto Hx_ = Hx.mutable_unchecked<3>();
    auto Hy_ = Hy.mutable_unchecked<3>();
    auto Hz_ = Hz.mutable_unchecked<3>();
    auto Phi1_ = Phi1.mutable_unchecked<4>();
    auto Phi2_ = Phi2.mutable_unchecked<4>();
    auto RA_ = RA.unchecked<2>();
    auto RB_ = RB.unchecked<2>();
    auto RE_ = RE.unchecked<2>();
    auto RF_ = RF.unchecked<2>();

    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    float dy = d;

    omp_set_num_threads(nthreads);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nx; ++i) {
        int ii = i + xs;
        for (int j = 0; j < ny; ++j) {
            int jj = yf - (j + 1);
            float RA01 = RA_(0, j) - 1;
            float RB0 = RB_(0, j);
            float RE0 = RE_(0, j);
            float RF0 = RF_(0, j);
            for (int k = 0; k < nz; ++k) {
                int kk = k + zs;

                // Hx
                int materialHx = ID_(3, ii, jj, kk);
                float dEz = (Ez_(ii, jj + 1, kk) - Ez_(ii, jj, kk)) / dy;
                Hx_(ii, jj, kk) -= updatecoeffsH_(materialHx, 4) * (RA01 * dEz + RB0 * Phi1_(0, i, j, k));
                Phi1_(0, i, j, k) = RE0 * Phi1_(0, i, j, k) - RF0 * dEz;

                // Hz
                int materialHz = ID_(5, ii, jj, kk);
                float dEx = (Ex_(ii, jj + 1, kk) - Ex_(ii, jj, kk)) / dy;
                Hz_(ii, jj, kk) += updatecoeffsH_(materialHz, 4) * (RA01 * dEx + RB0 * Phi2_(0, i, j, k));
                Phi2_(0, i, j, k) = RE0 * Phi2_(0, i, j, k) - RF0 * dEx;
            }
        }
    }
}

void order1_yplus(
    int xs,
    int xf,
    int ys,
    int yf,
    int zs,
    int zf,
    int nthreads,
    py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsH,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ID,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ex,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ey,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ez,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hx,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hy,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hz,
    py::array_t<float, py::array::c_style | py::array::forcecast> Phi1,
    py::array_t<float, py::array::c_style | py::array::forcecast> Phi2,
    py::array_t<float, py::array::c_style | py::array::forcecast> RA,
    py::array_t<float, py::array::c_style | py::array::forcecast> RB,
    py::array_t<float, py::array::c_style | py::array::forcecast> RE,
    py::array_t<float, py::array::c_style | py::array::forcecast> RF,
    float d
) {
    auto updatecoeffsH_ = updatecoeffsH.unchecked<2>();
    auto ID_ = ID.unchecked<4>();
    auto Ex_ = Ex.mutable_unchecked<3>();
    auto Ey_ = Ey.mutable_unchecked<3>();
    auto Ez_ = Ez.mutable_unchecked<3>();
    auto Hx_ = Hx.mutable_unchecked<3>();
    auto Hy_ = Hy.mutable_unchecked<3>();
    auto Hz_ = Hz.mutable_unchecked<3>();
    auto Phi1_ = Phi1.mutable_unchecked<4>();
    auto Phi2_ = Phi2.mutable_unchecked<4>();
    auto RA_ = RA.unchecked<2>();
    auto RB_ = RB.unchecked<2>();
    auto RE_ = RE.unchecked<2>();
    auto RF_ = RF.unchecked<2>();

    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;
    float dy = d;

    omp_set_num_threads(nthreads);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nx; ++i) {
        int ii = i + xs;
        for (int j = 0; j < ny; ++j) {
            int jj = j + ys;
            float RA01 = RA_(0, j) - 1;
            float RB0 = RB_(0, j);
            float RE0 = RE_(0, j);
            float RF0 = RF_(0, j);
            for (int k = 0; k < nz; ++k) {
                int kk = k + zs;
                // Hx
                int materialHx = ID_(3, ii, jj, kk);
                float dEz = (Ez_(ii, jj + 1, kk) - Ez_(ii, jj, kk)) / dy;
                Hx_(ii, jj, kk) -= updatecoeffsH_(materialHx, 4) * (RA01 * dEz + RB0 * Phi1_(0, i, j, k));
                Phi1_(0, i, j, k) = RE0 * Phi1_(0, i, j, k) - RF0 * dEz;
                // Hz
                int materialHz = ID_(5, ii, jj, kk);
                float dEx = (Ex_(ii, jj + 1, kk) - Ex_(ii, jj, kk)) / dy;
                Hz_(ii, jj, kk) += updatecoeffsH_(materialHz, 4) * (RA01 * dEx + RB0 * Phi2_(0, i, j, k));
                Phi2_(0, i, j, k) = RE0 * Phi2_(0, i, j, k) - RF0 * dEx;
            }
        }
    }
}

void order1_zminus(
    int xs,
    int xf,
    int ys,
    int yf,
    int zs,
    int zf,
    int nthreads,
    py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsH,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ID,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ex,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ey,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ez,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hx,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hy,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hz,
    py::array_t<float, py::array::c_style | py::array::forcecast> Phi1,
    py::array_t<float, py::array::c_style | py::array::forcecast> Phi2,
    py::array_t<float, py::array::c_style | py::array::forcecast> RA,
    py::array_t<float, py::array::c_style | py::array::forcecast> RB,
    py::array_t<float, py::array::c_style | py::array::forcecast> RE,
    py::array_t<float, py::array::c_style | py::array::forcecast> RF,
    float d
) {
    auto updatecoeffsH_ = updatecoeffsH.unchecked<2>();
    auto ID_ = ID.unchecked<4>();
    auto Ex_ = Ex.mutable_unchecked<3>();
    auto Ey_ = Ey.mutable_unchecked<3>();
    auto Ez_ = Ez.mutable_unchecked<3>();
    auto Hx_ = Hx.mutable_unchecked<3>();
    auto Hy_ = Hy.mutable_unchecked<3>();
    auto Hz_ = Hz.mutable_unchecked<3>();
    auto Phi1_ = Phi1.mutable_unchecked<4>();
    auto Phi2_ = Phi2.mutable_unchecked<4>();
    auto RA_ = RA.unchecked<2>();
    auto RB_ = RB.unchecked<2>();
    auto RE_ = RE.unchecked<2>();
    auto RF_ = RF.unchecked<2>();

    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    omp_set_num_threads(nthreads);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nx; ++i) {
        int ii = i + xs;
        for (int j = 0; j < ny; ++j) {
            int jj = j + ys;
            for (int k = 0; k < nz; ++k) {
                int kk = zf - (k + 1);
                float RA01 = RA_(0, k) - 1;
                float RB0 = RB_(0, k);
                float RE0 = RE_(0, k);
                float RF0 = RF_(0, k);

                // Hx
                int materialHx = ID_(3, ii, jj, kk);
                float dEy = (Ey_(ii, jj, kk + 1) - Ey_(ii, jj, kk)) / d;
                Hx_(ii, jj, kk) = (Hx_(ii, jj, kk) + updatecoeffsH_(materialHx, 4) *
                                  (RA01 * dEy + RB0 * Phi1_(0, i, j, k)));
                Phi1_(0, i, j, k) = RE0 * Phi1_(0, i, j, k) - RF0 * dEy;

                // Hy
                int materialHy = ID_(4, ii, jj, kk);
                float dEx = (Ex_(ii, jj, kk + 1) - Ex_(ii, jj, kk)) / d;
                Hy_(ii, jj, kk) = (Hy_(ii, jj, kk) - updatecoeffsH_(materialHy, 4) *
                                  (RA01 * dEx + RB0 * Phi2_(0, i, j, k)));
                Phi2_(0, i, j, k) = RE0 * Phi2_(0, i, j, k) - RF0 * dEx;
            }
        }
    }
}

void order1_zplus(
    int xs,
    int xf,
    int ys,
    int yf,
    int zs,
    int zf,
    int nthreads,
    py::array_t<float, py::array::c_style | py::array::forcecast> updatecoeffsH,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ID,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ex,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ey,
    py::array_t<float, py::array::c_style | py::array::forcecast> Ez,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hx,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hy,
    py::array_t<float, py::array::c_style | py::array::forcecast> Hz,
    py::array_t<float, py::array::c_style | py::array::forcecast> Phi1,
    py::array_t<float, py::array::c_style | py::array::forcecast> Phi2,
    py::array_t<float, py::array::c_style | py::array::forcecast> RA,
    py::array_t<float, py::array::c_style | py::array::forcecast> RB,
    py::array_t<float, py::array::c_style | py::array::forcecast> RE,
    py::array_t<float, py::array::c_style | py::array::forcecast> RF,
    float d
) {
    auto updatecoeffsH_ = updatecoeffsH.unchecked<2>();
    auto ID_ = ID.unchecked<4>();
    auto Ex_ = Ex.mutable_unchecked<3>();
    auto Ey_ = Ey.mutable_unchecked<3>();
    auto Ez_ = Ez.mutable_unchecked<3>();
    auto Hx_ = Hx.mutable_unchecked<3>();
    auto Hy_ = Hy.mutable_unchecked<3>();
    auto Hz_ = Hz.mutable_unchecked<3>();
    auto Phi1_ = Phi1.mutable_unchecked<4>();
    auto Phi2_ = Phi2.mutable_unchecked<4>();
    auto RA_ = RA.unchecked<2>();
    auto RB_ = RB.unchecked<2>();
    auto RE_ = RE.unchecked<2>();
    auto RF_ = RF.unchecked<2>();

    int nx = xf - xs;
    int ny = yf - ys;
    int nz = zf - zs;

    omp_set_num_threads(nthreads);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nx; ++i) {
        int ii = i + xs;
        for (int j = 0; j < ny; ++j) {
            int jj = j + ys;
            for (int k = 0; k < nz; ++k) {
                int kk = k + zs;
                float RA01 = RA_(0, k) - 1;
                float RB0 = RB_(0, k);
                float RE0 = RE_(0, k);
                float RF0 = RF_(0, k);

                // Hx
                int materialHx = ID_(3, ii, jj, kk);
                float dEy = (Ey_(ii, jj, kk + 1) - Ey_(ii, jj, kk)) / d;
                Hx_(ii, jj, kk) += updatecoeffsH_(materialHx, 4) * (RA01 * dEy + RB0 * Phi1_(0, i, j, k));
                Phi1_(0, i, j, k) = RE0 * Phi1_(0, i, j, k) - RF0 * dEy;

                // Hy
                int materialHy = ID_(4, ii, jj, kk);
                float dEx = (Ex_(ii, jj, kk + 1) - Ex_(ii, jj, kk)) / d;
                Hy_(ii, jj, kk) -= updatecoeffsH_(materialHy, 4) * (RA01 * dEx + RB0 * Phi2_(0, i, j, k));
                Phi2_(0, i, j, k) = RE0 * Phi2_(0, i, j, k) - RF0 * dEx;
            }
        }
    }
}



PYBIND11_MODULE(pybind11_pml_updates_magnetic_HORIPML, m) {
    m.def("order1_xminus", &order1_xminus, py::arg("xs"), py::arg("xf"), py::arg("ys"), py::arg("yf"),
          py::arg("zs"), py::arg("zf"), py::arg("nthreads"), py::arg("updatecoeffsH"), py::arg("ID"),
          py::arg("Ex"), py::arg("Ey"), py::arg("Ez"), py::arg("Hx"), py::arg("Hy"), py::arg("Hz"),
          py::arg("Phi1"), py::arg("Phi2"), py::arg("RA"), py::arg("RB"), py::arg("RE"), py::arg("RF"),
          py::arg("d"));
    m.def("order1_xplus", &order1_xplus, py::arg("xs"), py::arg("xf"), py::arg("ys"), py::arg("yf"),
          py::arg("zs"), py::arg("zf"), py::arg("nthreads"), py::arg("updatecoeffsH"), py::arg("ID"),
          py::arg("Ex"), py::arg("Ey"), py::arg("Ez"), py::arg("Hx"), py::arg("Hy"), py::arg("Hz"),
          py::arg("Phi1"), py::arg("Phi2"), py::arg("RA"), py::arg("RB"), py::arg("RE"), py::arg("RF"),
          py::arg("d"));
    m.def("order1_yminus", &order1_yminus, py::arg("xs"), py::arg("xf"), py::arg("ys"), py::arg("yf"),
          py::arg("zs"), py::arg("zf"), py::arg("nthreads"), py::arg("updatecoeffsH"), py::arg("ID"),
          py::arg("Ex"), py::arg("Ey"), py::arg("Ez"), py::arg("Hx"), py::arg("Hy"), py::arg("Hz"),
          py::arg("Phi1"), py::arg("Phi2"), py::arg("RA"), py::arg("RB"), py::arg("RE"), py::arg("RF"),
          py::arg("d"));
    m.def("order1_yplus", &order1_yplus, py::arg("xs"), py::arg("xf"), py::arg("ys"), py::arg("yf"),
          py::arg("zs"), py::arg("zf"), py::arg("nthreads"), py::arg("updatecoeffsH"), py::arg("ID"),
          py::arg("Ex"), py::arg("Ey"), py::arg("Ez"), py::arg("Hx"), py::arg("Hy"), py::arg("Hz"),
          py::arg("Phi1"), py::arg("Phi2"), py::arg("RA"), py::arg("RB"), py::arg("RE"), py::arg("RF"),
          py::arg("d"));
    m.def("order1_zminus", &order1_zminus, py::arg("xs"), py::arg("xf"), py::arg("ys"), py::arg("yf"),
          py::arg("zs"), py::arg("zf"), py::arg("nthreads"), py::arg("updatecoeffsH"), py::arg("ID"),
          py::arg("Ex"), py::arg("Ey"), py::arg("Ez"), py::arg("Hx"), py::arg("Hy"), py::arg("Hz"),
          py::arg("Phi1"), py::arg("Phi2"), py::arg("RA"), py::arg("RB"), py::arg("RE"), py::arg("RF"),
          py::arg("d"));
    m.def("order1_zplus", &order1_zplus, py::arg("xs"), py::arg("xf"), py::arg("ys"), py::arg("yf"),
          py::arg("zs"), py::arg("zf"), py::arg("nthreads"), py::arg("updatecoeffsH"), py::arg("ID"),
          py::arg("Ex"), py::arg("Ey"), py::arg("Ez"), py::arg("Hx"), py::arg("Hy"), py::arg("Hz"),
          py::arg("Phi1"), py::arg("Phi2"), py::arg("RA"), py::arg("RB"), py::arg("RE"), py::arg("RF"),
          py::arg("d"));
}