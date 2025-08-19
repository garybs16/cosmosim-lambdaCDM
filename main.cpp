// bigbang3d_pretty.cpp
// 3D N-body toy + "beauty" renderer (soft splats, depth fog, velocity color)
// Optional red–cyan anaglyph stereo output.
// C++17, MSVC/GCC/Clang. No external deps.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

#ifdef _WIN32
#include <direct.h>
static inline void make_dir(const std::string& p) { _mkdir(p.c_str()); }
#else
#include <sys/stat.h>
#include <sys/types.h>
static inline void make_dir(const std::string& p) { mkdir(p.c_str(), 0755); }
#endif

static constexpr double PI = 3.141592653589793238462643383279502884;

struct Vec3 {
    double x = 0, y = 0, z = 0;
    Vec3() = default; Vec3(double X, double Y, double Z) :x(X), y(Y), z(Z) {}
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(double s) { x *= s; y *= s; z *= s; return *this; }
};
static inline Vec3 operator+(Vec3 a, const Vec3& b) { return a += b; }
static inline Vec3 operator-(Vec3 a, const Vec3& b) { return a -= b; }
static inline Vec3 operator*(Vec3 a, double s) { return a *= s; }
static inline Vec3 operator*(double s, Vec3 a) { return a *= s; }
static inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline double norm2(const Vec3& a) { return dot(a, a); }

static inline double minimg1(double d, double L) { if (d > 0.5 * L) d -= L; if (d < -0.5 * L) d += L; return d; }
static inline Vec3 minimg(Vec3 d, double L) { d.x = minimg1(d.x, L); d.y = minimg1(d.y, L); d.z = minimg1(d.z, L); return d; }
static inline void wrap(Vec3& p, double L) {
    p.x = p.x - std::floor(p.x / L) * L;
    p.y = p.y - std::floor(p.y / L) * L;
    p.z = p.z - std::floor(p.z / L) * L;
}

struct Particle { Vec3 x; Vec3 v; };

static void save_ppm(const std::string& fname, int W, int H, const std::vector<unsigned char>& rgb) {
    std::ofstream f(fname, std::ios::binary);
    if (!f) { std::cerr << "Failed to write " << fname << "\n"; return; }
    f << "P6\n" << W << " " << H << "\n255\n";
    f.write(reinterpret_cast<const char*>(rgb.data()), rgb.size());
}

// ---------------- Camera ----------------
struct Camera {
    double cx = 0, cy = 0, cz = 0;   // position
    double tx = 0, ty = 0, tz = 0;   // look-at
    double upx = 0, upy = 1, upz = 0;
    double fov_deg = 55.0, near_z = 0.05;

    void basis(Vec3& u, Vec3& v, Vec3& w) const {
        Vec3 f{ tx - cx, ty - cy, tz - cz };
        double fl2 = norm2(f);
        if (fl2 < 1e-30) f = { 0,0,1 }; else f = (1.0 / std::sqrt(fl2)) * f;
        w = { -f.x,-f.y,-f.z };
        Vec3 up{ upx,upy,upz };
        Vec3 cu{ up.y * w.z - up.z * w.y, up.z * w.x - up.x * w.z, up.x * w.y - up.y * w.x };
        double cul2 = norm2(cu);
        Vec3 u_;
        if (cul2 < 1e-30) u_ = { 1,0,0 }; else u_ = (1.0 / std::sqrt(cul2)) * cu;
        u = u_;
        v = { w.y * u.z - w.z * u.y, w.z * u.x - w.x * u.z, w.x * u.y - w.y * u.x };
    }
    Vec3 to_cam(const Vec3& p) const {
        Vec3 u, v, w; basis(u, v, w);
        Vec3 d{ p.x - cx, p.y - cy, p.z - cz };
        return { dot(d,u), dot(d,v), dot(d,w) }; // right, up, forward(-z)
    }
};

// ---- small helpers for pretty rendering ----
static inline unsigned char clamp255(int v) { return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v)); }
static inline void add_rgb(std::vector<unsigned char>& img, int W, int H, int px, int py, int r, int g, int b) {
    if (px < 0 || px >= W || py < 0 || py >= H) return;
    size_t idx = (size_t)(py * W + px) * 3;
    int R = img[idx + 0] + r, G = img[idx + 1] + g, B = img[idx + 2] + b;
    img[idx + 0] = clamp255(R); img[idx + 1] = clamp255(G); img[idx + 2] = clamp255(B);
}

// Map speed to color (slow=warm/amber, fast=cyan/blue).
static inline void speed_to_rgb(double speed, double smax, int& R, int& G, int& B) {
    double t = std::min(1.0, std::max(0.0, speed / (1e-12 + smax)));
    // simple 2-stop gradient: amber (255,180,60) -> cyan (80,220,255)
    double r = (1.0 - t) * 255.0 + t * 80.0;
    double g = (1.0 - t) * 180.0 + t * 220.0;
    double b = (1.0 - t) * 60.0 + t * 255.0;
    R = (int)std::lround(r); G = (int)std::lround(g); B = (int)std::lround(b);
}

// Soft splat (Gaussian-ish) with size ∝ 1/z and depth fog.
static void render_beauty(
    const std::vector<Particle>& P, double L,
    int frame_index, int total_frames,
    int W, int H, const std::string& fname)
{
    // Orbit camera
    const double angle = (2.0 * PI) * (double)frame_index / std::max(1, total_frames);
    const Vec3 center{ L * 0.5, L * 0.5, L * 0.5 };
    const double radius = 1.6 * L, height = 0.5 * L;

    Camera cam;
    cam.cx = center.x + radius * std::cos(angle);
    cam.cy = center.y + height;
    cam.cz = center.z + radius * std::sin(angle);
    cam.tx = center.x; cam.ty = center.y; cam.tz = center.z;
    cam.fov_deg = 55.0; cam.near_z = 0.05;

    const double fov_rad = cam.fov_deg * PI / 180.0;
    const double fy = 0.5 * H / std::tan(0.5 * fov_rad);
    const double fx = fy;

    std::vector<unsigned char> img((size_t)W * H * 3, 0);

    // find a rough speed max for coloring
    double smax = 0.0;
    for (const auto& p : P) smax = std::max(smax, std::sqrt(norm2(p.v)));
    if (smax <= 1e-12) smax = 1.0;

    for (const auto& part : P) {
        const Vec3 pc = cam.to_cam(part.x);
        const double z = -pc.z;
        if (z <= cam.near_z) continue;

        const double x_pix = (pc.x * fx) / z;
        const double y_pix = (pc.y * fy) / z;
        const int cx = (int)std::lround(W * 0.5 + x_pix);
        const int cy = (int)std::lround(H * 0.5 - y_pix);

        // splat radius (in pixels) scales with closeness
        double r_px = 1.5 + 10.0 * std::min(1.0, 1.0 / (0.2 + z / (1.5 * L)));
        int rad = (int)std::ceil(r_px);

        // base color by speed
        int baseR, baseG, baseB;
        speed_to_rgb(std::sqrt(norm2(part.v)), smax, baseR, baseG, baseB);

        // depth fog factor (farther = dimmer)
        double fog = std::max(0.25, std::min(1.0, 1.0 / (0.1 + z / (1.6 * L))));

        // draw soft disk
        double inv2s2 = 1.0 / (2.0 * (0.35 * rad) * (0.35 * rad));
        for (int dy = -rad; dy <= rad; ++dy) {
            for (int dx = -rad; dx <= rad; ++dx) {
                double rr = dx * dx + dy * dy;
                if (rr > (rad * rad)) continue;
                double w = std::exp(-rr * inv2s2) * fog; // soft falloff + fog
                int r = (int)std::lround(baseR * w);
                int g = (int)std::lround(baseG * w);
                int b = (int)std::lround(baseB * w);
                add_rgb(img, W, H, cx + dx, cy + dy, r, g, b);
            }
        }
    }

    save_ppm(fname, W, H, img);
}

// Red–cyan stereo anaglyph (needs cheap 3D glasses). Offset two cameras.
static void render_anaglyph(
    const std::vector<Particle>& P, double L,
    int frame_index, int total_frames,
    int W, int H, const std::string& fname)
{
    const double angle = (2.0 * PI) * (double)frame_index / std::max(1, total_frames);
    const Vec3 center{ L * 0.5, L * 0.5, L * 0.5 };
    const double radius = 1.6 * L, height = 0.5 * L;

    Camera base;
    base.cx = center.x + radius * std::cos(angle);
    base.cy = center.y + height;
    base.cz = center.z + radius * std::sin(angle);
    base.tx = center.x; base.ty = center.y; base.tz = center.z;
    base.fov_deg = 55.0; base.near_z = 0.05;

    // eye separation ~ a few percent of L along camera-right axis
    Vec3 u, v, w; base.basis(u, v, w);
    double eye = 0.04 * L;

    Camera left = base;  left.cx -= eye * u.x; left.cy -= eye * u.y; left.cz -= eye * u.z;
    Camera right = base; right.cx += eye * u.x; right.cy += eye * u.y; right.cz += eye * u.z;

    const double fov_rad = base.fov_deg * PI / 180.0;
    const double fy = 0.5 * H / std::tan(0.5 * fov_rad);
    const double fx = fy;

    std::vector<unsigned char> img((size_t)W * H * 3, 0);

    auto splat_eye = [&](const Camera& cam, int chanR, int chanG, int chanB) {
        // rough speed max for coloring
        double smax = 0.0; for (const auto& p : P) smax = std::max(smax, std::sqrt(norm2(p.v))); if (smax <= 1e-12) smax = 1.0;
        for (const auto& part : P) {
            Vec3 pc = cam.to_cam(part.x);
            double z = -pc.z; if (z <= cam.near_z) continue;
            double x_pix = (pc.x * fx) / z;
            double y_pix = (pc.y * fy) / z;
            int cx = (int)std::lround(W * 0.5 + x_pix);
            int cy = (int)std::lround(H * 0.5 - y_pix);
            double r_px = 1.2 + 8.0 * std::min(1.0, 1.0 / (0.2 + z / (1.5 * L)));
            int rad = (int)std::ceil(r_px);
            int baseR, baseG, baseB; speed_to_rgb(std::sqrt(norm2(part.v)), smax, baseR, baseG, baseB);
            double fog = std::max(0.25, std::min(1.0, 1.0 / (0.1 + z / (1.6 * L))));
            double inv2s2 = 1.0 / (2.0 * (0.35 * rad) * (0.35 * rad));
            for (int dy = -rad; dy <= rad; ++dy) {
                for (int dx = -rad; dx <= rad; ++dx) {
                    double rr = dx * dx + dy * dy; if (rr > rad * rad) continue;
                    double w = std::exp(-rr * inv2s2) * fog;
                    int r = (int)std::lround(baseR * w);
                    int g = (int)std::lround(baseG * w);
                    int b = (int)std::lround(baseB * w);
                    int px = cx + dx, py = cy + dy; if (px < 0 || px >= W || py < 0 || py >= H) continue;
                    size_t idx = (size_t)(py * W + px) * 3;
                    // Write only selected channels (R for left, G+B for right)
                    if (chanR) img[idx + 0] = clamp255(img[idx + 0] + r);
                    if (chanG) img[idx + 1] = clamp255(img[idx + 1] + g);
                    if (chanB) img[idx + 2] = clamp255(img[idx + 2] + b);
                }
            }
        }
        };

    // Left eye contributes RED; right contributes CYAN
    splat_eye(left, 1, 0, 0);
    splat_eye(right, 0, 1, 1);

    save_ppm(fname, W, H, img);
}

// ---------------- Simulation ----------------
int main(int argc, char** argv) {
    // CLI: N steps dump_every mode stereo
    int    N = 4000;
    int    steps = 2400;
    int    dump_every = 4;
    int    mode = 0;     // 0=beauty, 1=anaglyph stereo
    // physics
    const double L = 1.0, G = 1.0, soft = 0.0025;
    const double a_start = 0.02, a_end = 1.0, H0 = 1.0;
    const double jitter = 0.004, sigma_v = 0.03;
    const unsigned seed = 1337;
    // render
    const int W = 1280, H = 720;
    const std::string outdir = "frames";

    if (argc > 1) N = std::max(200, std::atoi(argv[1]));
    if (argc > 2) steps = std::max(200, std::atoi(argv[2]));
    if (argc > 3) dump_every = std::max(1, std::atoi(argv[3]));
    if (argc > 4) mode = std::max(0, std::min(1, std::atoi(argv[4])));

    make_dir(outdir);

    std::vector<Particle> P((size_t)N);
    std::vector<Vec3> acc((size_t)N);
    const double eps2 = soft * soft;
    const double invN = 1.0 / (double)N;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(-0.35, 0.35);
    std::normal_distribution<double> NV(0.0, 1.0);

    const int side = (int)std::ceil(std::cbrt((double)N));
    int idx = 0;
    for (int k = 0; k < side && idx < N; ++k)
        for (int j = 0; j < side && idx < N; ++j)
            for (int i = 0; i < side && idx < N; ++i) {
                Vec3 p{ (i + 0.5) * (L / side), (j + 0.5) * (L / side), (k + 0.5) * (L / side) };
                p.x += jitter * U(rng); p.y += jitter * U(rng); p.z += jitter * U(rng); wrap(p, L);
                Vec3 v{ sigma_v * NV(rng), sigma_v * NV(rng), sigma_v * NV(rng) };
                P[idx++] = { p, v };
            }

    auto H_of_a = [&](double a) { return H0 * std::pow(a, -1.5); };
    auto dt_from_da = [&](double a, double da) { return da / (a * H_of_a(a)); };

    double a = a_start;
    const double da = (a_end - a_start) / (double)steps;

    // initial frame
    {
        std::ostringstream name; name << outdir << "/frame_" << std::setw(5) << std::setfill('0') << 0 << ".ppm";
        if (mode == 0) render_beauty(P, L, 0, steps / dump_every + 1, W, H, name.str());
        else        render_anaglyph(P, L, 0, steps / dump_every + 1, W, H, name.str());
    }
    int frame_id = 1, total_frames = steps / dump_every + 1;

    for (int s = 1; s <= steps; ++s) {
        const double dt = dt_from_da(a, da);
        const double Hh = H_of_a(a);

        std::fill(acc.begin(), acc.end(), Vec3{ 0,0,0 });
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                Vec3 d = minimg(P[i].x - P[j].x, L);
                double r2 = norm2(d) + eps2;
                double inv_r = 1.0 / std::sqrt(r2);
                double inv_r3 = inv_r / r2;
                double coef = (G * invN) * inv_r3 / (a * a * a);
                Vec3 aij = (-coef) * d;
                acc[i] += aij; acc[j] -= aij;
            }
        }
        for (int i = 0; i < N; ++i) {
            P[i].v.x += dt * (acc[i].x - 2.0 * Hh * P[i].v.x);
            P[i].v.y += dt * (acc[i].y - 2.0 * Hh * P[i].v.y);
            P[i].v.z += dt * (acc[i].z - 2.0 * Hh * P[i].v.z);
            P[i].x.x += dt * P[i].v.x; P[i].x.y += dt * P[i].v.y; P[i].x.z += dt * P[i].v.z;
            wrap(P[i].x, L);
        }
        a += da;

        if (s % dump_every == 0) {
            std::ostringstream name; name << outdir << "/frame_" << std::setw(5) << std::setfill('0') << frame_id << ".ppm";
            if (mode == 0) render_beauty(P, L, frame_id, total_frames, W, H, name.str());
            else        render_anaglyph(P, L, frame_id, total_frames, W, H, name.str());
            if ((frame_id % 30) == 0) {
                std::cout << "a=" << std::fixed << std::setprecision(4) << a
                    << "  step " << s << "/" << steps
                    << "  frame " << frame_id << "/" << total_frames << "\n";
            }
            ++frame_id;
        }
    }

    std::cout << "Done. Frames in ./frames\n"
        << "Video: ffmpeg -framerate 30 -i frames/frame_%05d.ppm -pix_fmt yuv420p bigbang3d.mp4\n"
        << "Stereo (mode=1) works with red–cyan 3D glasses.\n";
    return 0;
}
