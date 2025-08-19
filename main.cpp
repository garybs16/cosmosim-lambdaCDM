// cosmosim.cpp
// ------------------------------------------------------------
// "CosmoSim-ΛCDM": A serious starting point for a scientific 3D N-body code
// ------------------------------------------------------------
// Features
//  - 3D particles in a periodic cubic box (minimum-image wrapping)
//  - ΛCDM expansion: H(a)=H0*sqrt(Ωm a^-3 + Ωr a^-4 + ΩΛ) [Ωr optional]
//  - Comoving coordinates x in [0,L), peculiar velocity v = dx/dt
//  - Equation of motion: d^2x/dt^2 + 2H dx/dt = -(1/a^3) ∇_x Φ
//  - Barnes–Hut octree O(N log N) gravity with Plummer softening
//  - KDK leapfrog in physical time via constant Δa steps (dt = da/(a H(a)))
//  - OpenMP parallel force evaluation
//  - Snapshot CSV output (positions, velocities, a)
//
// What this is NOT (yet): full TreePM/Ewald periodic solver or ICs from P(k).
// It is, however, an actually-usable base you can extend to TreePM + FFTW, HDF5, MPI.
// ------------------------------------------------------------
// Build (Linux/macOS, Clang or GCC):
//   g++ -O3 -march=native -fopenmp -std=c++20 cosmosim.cpp -o cosmosim
// Run (defaults shown):
//   ./cosmosim [N=20000] [steps=3000] [theta=0.6] [soft=L*4e-4] [L=100.0]
//              [a_start=0.02] [a_end=1.0] [dump_every=50]
// Example:
//   ./cosmosim 100000 4000 0.6 0.04 200 0.02 1.0 40   # (soft=0.04 means 0.04 units)
// Snapshots go to ./snapshots/a_XXXXXX.csv (XXXXXX = a*1e4 rounded)
// Stitch preview with: ffmpeg -framerate 30 -pattern_type glob -i 'snapshots/*.csv' ...
// (Better: view as points in ParaView or use a small Python viz.)
// ------------------------------------------------------------
// DISCLAIMER: BH + minimum-image is an approximation for periodic gravity.
// For publication-grade long-range periodic forces, add a PM mesh (TreePM) or Ewald.
// ------------------------------------------------------------

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------- Math ----------------------------------
struct Vec3 {
    double x{0}, y{0}, z{0};
    Vec3() = default;
    Vec3(double X,double Y,double Z):x(X),y(Y),z(Z){}
    Vec3& operator+=(const Vec3& b){ x+=b.x; y+=b.y; z+=b.z; return *this; }
    Vec3& operator-=(const Vec3& b){ x-=b.x; y-=b.y; z-=b.z; return *this; }
    Vec3& operator*=(double s){ x*=s; y*=s; z*=s; return *this; }
};
static inline Vec3 operator+(Vec3 a,const Vec3& b){ return a+=b; }
static inline Vec3 operator-(Vec3 a,const Vec3& b){ return a-=b; }
static inline Vec3 operator*(Vec3 a,double s){ return a*=s; }
static inline Vec3 operator*(double s,Vec3 a){ return a*=s; }
static inline double dot(const Vec3& a,const Vec3& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline double norm2(const Vec3& a){ return dot(a,a); }

// periodic minimal image in [0,L)
static inline double min_image_1d(double dx, double L){
    if(dx >  0.5*L) dx -= L;
    if(dx < -0.5*L) dx += L;
    return dx;
}
static inline Vec3 min_image(Vec3 d, double L){
    d.x = min_image_1d(d.x, L);
    d.y = min_image_1d(d.y, L);
    d.z = min_image_1d(d.z, L);
    return d;
}

// -------------------------------- Cosmology --------------------------------
struct Cosmology {
    double H0   = 1.0;   // code units
    double Omega_m = 0.315;
    double Omega_L = 0.685; // flat if Omega_m+Omega_L+Omega_r==1
    double Omega_r = 0.0;   // usually tiny at late times; keep 0 by default

    double H_of_a(double a) const {
        // H(a) = H0 * sqrt(Ωm a^-3 + Ωr a^-4 + ΩΛ)
        return H0 * std::sqrt(Omega_m*std::pow(a,-3.0) + Omega_r*std::pow(a,-4.0) + Omega_L);
    }

    double dt_from_da(double a, double da) const {
        // da/dt = a * H(a) => dt = da / (a * H(a))
        return da / (a * H_of_a(a));
    }
};

// --------------------------------- Bodies ---------------------------------
struct Body { Vec3 x; Vec3 v; double m; };

// --------------------------------- Octree ---------------------------------
struct Node {
    Vec3 center;     // cube center in [0,L)
    double half;     // half-width
    double mass{0};  // total mass of node
    Vec3 com{0,0,0}; // center of mass in [0,L)
    int child[8];    // indices into pool, -1 if empty
    int first{-1};   // body index if leaf with single body (we'll allow small leaf fanout if desired)
    int count{0};    // number of bodies in subtree
    bool leaf{true};
};

struct Octree {
    std::vector<Node> pool;
    double L{1.0};
    const std::vector<Body>* bodies{nullptr};

    int new_node(const Vec3& c, double h){
        Node n; n.center=c; n.half=h; n.mass=0; n.com={0,0,0}; n.leaf=true; n.first=-1; n.count=0; for(int i=0;i<8;++i)n.child[i]=-1; pool.push_back(n); return (int)pool.size()-1;
    }

    int octant(const Vec3& p, const Vec3& c){
        int o=0; if(p.x>=c.x) o|=1; if(p.y>=c.y) o|=2; if(p.z>=c.z) o|=4; return o; }

    Vec3 child_center(const Vec3& c, double h, int o){
        return Vec3{ c.x + ((o&1)? +h : -h)/2, c.y + ((o&2)? +h : -h)/2, c.z + ((o&4)? +h : -h)/2 };
    }

    void insert(int idx, int bidx){
        Node& n = pool[idx];
        if(n.leaf){
            if(n.count==0){ n.first=bidx; n.count=1; return; }
            // split
            int old = n.first;
            n.leaf=false; n.first=-1; // create children lazily
            for(int i=0;i<8;++i){
                Vec3 cc = child_center(n.center, 2*n.half, i);
                n.child[i] = new_node(cc, n.half);
            }
            // reinsert existing
            int o1 = octant((*bodies)[old].x, n.center);
            insert(n.child[o1], old);
            // insert new
            int o2 = octant((*bodies)[bidx].x, n.center);
            insert(n.child[o2], bidx);
            n.count=2;
        } else {
            int o = octant((*bodies)[bidx].x, n.center);
            insert(n.child[o], bidx);
            n.count++;
        }
    }

    void build(const std::vector<Body>& B, double boxL){
        L = boxL; bodies = &B; pool.clear(); pool.reserve(B.size()*2);
        // root spans entire [0,L)^3 with center at (L/2,L/2,L/2)
        int root = new_node({L*0.5,L*0.5,L*0.5}, L*0.5);
        for(int i=0;i<(int)B.size();++i) insert(root, i);
        // accumulate mass and com
        accumulate(root);
    }

    void accumulate(int idx){
        Node& n = pool[idx];
        if(n.leaf){
            if(n.count==1){
                const Body& b = (*bodies)[n.first];
                n.mass = b.m; n.com = b.x; return;
            } else { n.mass=0; n.com={0,0,0}; return; }
        }
        double M=0; Vec3 C{0,0,0};
        for(int i=0;i<8;++i){ int c = n.child[i]; if(c<0) continue; accumulate(c); const Node& ch=pool[c]; if(ch.mass<=0) continue; M += ch.mass; // COM in periodic box: accumulate using minimum-image relative to current C guess
            if(M==ch.mass){ C = ch.com; }
            else {
                Vec3 d = min_image(ch.com - C, L); // move COM towards the child COM across boundaries if needed
                C += (ch.mass / M) * d; C.x = std::fmod(C.x+L,L); C.y = std::fmod(C.y+L,L); C.z = std::fmod(C.z+L,L);
            }
        }
        n.mass=M; n.com=C;
    }

    // gravitational acceleration contribution on body i from node idx
    Vec3 acc_from_node(int idx, int i, double theta, double eps2) const {
        const Node& n = pool[idx];
        if(n.mass<=0) return {0,0,0};
        const Vec3& xi = (*bodies)[i].x;
        Vec3 r = min_image(n.com - xi, L);
        double dist2 = norm2(r) + eps2;
        double dist = std::sqrt(dist2);
        double size = n.half*2; // node width
        if(n.leaf){
            if(n.count==0) return {0,0,0};
            if(n.count==1){
                if(n.first==i) return {0,0,0};
                // single particle approx
                double inv_r3 = 1.0/(dist2*dist);
                return (n.mass * inv_r3) * r;
            }
        }
        // Barnes–Hut acceptance criterion
        if(size / dist < theta){
            double inv_r3 = 1.0/(dist2*dist);
            return (n.mass * inv_r3) * r;
        } else {
            Vec3 a{0,0,0};
            for(int c=0;c<8;++c){ int ci=n.child[c]; if(ci<0) continue; a += acc_from_node(ci, i, theta, eps2); }
            return a;
        }
    }
};

// ------------------------------- Utilities --------------------------------
static inline void ensure_dir(const std::string& path){
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755);
#endif
}

static void write_csv(const std::string& fname, const std::vector<Body>& B, double a){
    std::ofstream f(fname);
    if(!f){ std::cerr << "Cannot write "<<fname<<"\n"; return; }
    f << "# a,"<< std::setprecision(10) << a << "\n";
    f << "x,y,z,vx,vy,vz\n";
    f.setf(std::ios::fixed); f<<std::setprecision(6);
    for(const auto& b: B){
        f<<b.x.x<<","<<b.x.y<<","<<b.x.z<<","<<b.v.x<<","<<b.v.y<<","<<b.v.z<<"\n";
    }
}

// ------------------------------- Simulation --------------------------------
struct Sim {
    // parameters
    int N = 20000;
    int steps = 3000;
    int dump_every = 50;
    double L = 100.0;         // box size (code units)
    double soft = 0.04;       // Plummer softening length (absolute units)
    double theta = 0.6;       // BH opening angle
    double G = 1.0;           // gravitational constant in code units
    double a_start = 0.02;
    double a_end   = 1.0;
    unsigned seed = 1337;

    Cosmology cosmo;

    // state
    std::vector<Body> B;
    std::vector<Vec3> acc;

    void init_bodies(){
        B.resize(N); acc.assign(N, {0,0,0});
        std::mt19937 rng(seed);
        // Quiet start on grid + jitter + tiny random velocities
        int side = (int)std::ceil(std::cbrt((double)N));
        int idx=0; double mass = 1.0 / (double)N;
        std::uniform_real_distribution<double> U(-0.35,0.35);
        std::normal_distribution<double> NV(0.0, 1e-3);
        for(int k=0;k<side && idx<N;++k){
            for(int j=0;j<side && idx<N;++j){
                for(int i=0;i<side && idx<N;++i){
                    Vec3 p{ (i+0.5)*(L/side), (j+0.5)*(L/side), (k+0.5)*(L/side) };
                    p.x += U(rng); p.y += U(rng); p.z += U(rng); // small jitter (absolute units)
                    // wrap
                    p.x = std::fmod(p.x+L, L); p.y = std::fmod(p.y+L, L); p.z = std::fmod(p.z+L, L);
                    Vec3 v{ NV(rng), NV(rng), NV(rng) }; // tiny peculiar vel
                    B[idx++] = {p, v, mass};
                }
            }
        }
    }

    void compute_accelerations(double a){
        Octree tree; tree.build(B, L);
        double eps2 = soft*soft;
        // Parallel over particles; tree is read-only
        #pragma omp parallel for schedule(static)
        for(int i=0;i<N;++i){
            Vec3 a_dimless = tree.acc_from_node(0, i, theta, eps2); // returns sum(m_j r_ij / |r|^3)
            // scale by G and 1/a^3 (comoving)
            acc[i] = (G / (a*a*a)) * a_dimless;
        }
    }

    void step_system(double& a, double da){
        double dt = cosmo.dt_from_da(a, da);
        double H = cosmo.H_of_a(a);
        double half = 0.5*dt;
        // Kick (half-step) with Hubble friction (explicit)
        for(int i=0;i<N;++i){
            B[i].v.x += half * (acc[i].x - 2.0*H*B[i].v.x);
            B[i].v.y += half * (acc[i].y - 2.0*H*B[i].v.y);
            B[i].v.z += half * (acc[i].z - 2.0*H*B[i].v.z);
        }
        // Drift
        for(int i=0;i<N;++i){
            B[i].x.x += dt * B[i].v.x; B[i].x.y += dt * B[i].v.y; B[i].x.z += dt * B[i].v.z;
            // wrap into [0,L)
            B[i].x.x = B[i].x.x - std::floor(B[i].x.x / L) * L;
            B[i].x.y = B[i].x.y - std::floor(B[i].x.y / L) * L;
            B[i].x.z = B[i].x.z - std::floor(B[i].x.z / L) * L;
        }
        // advance a
        a += da;
        // Recompute acc at new positions
        compute_accelerations(a);
        double Hn = cosmo.H_of_a(a);
        // Kick (half-step) with new acc & H
        for(int i=0;i<N;++i){
            B[i].v.x += half * (acc[i].x - 2.0*Hn*B[i].v.x);
            B[i].v.y += half * (acc[i].y - 2.0*Hn*B[i].v.y);
            B[i].v.z += half * (acc[i].z - 2.0*Hn*B[i].v.z);
        }
    }

    void run(){
        ensure_dir("snapshots");
        init_bodies();
        double a = a_start;
        double da = (a_end - a_start) / (double)steps;
        // initial forces
        compute_accelerations(a);

        auto t0 = std::chrono::high_resolution_clock::now();
        for(int s=0; s<steps; ++s){
            step_system(a, da);
            if((s % dump_every) == 0){
                int ia = (int)std::llround(a*10000.0);
                std::ostringstream name; name<<"snapshots/a_"<<std::setw(6)<<std::setfill('0')<<ia<<".csv";
                write_csv(name.str(), B, a);
            }
            if((s % 20)==0){
                auto t1=std::chrono::high_resolution_clock::now();
                double sec=std::chrono::duration<double>(t1-t0).count();
                std::cerr << "step "<<s<<"/"<<steps<<"  a="<<std::fixed<<std::setprecision(4)<<a
                          << "  t="<<std::setprecision(2)<<sec<<"s\r";
            }
        }
        std::cerr<<"\nDone. Snapshots in ./snapshots\n";
    }
};

int main(int argc, char** argv){
    Sim sim;
    // Defaults are set in struct; override from CLI
    if(argc>1) sim.N = std::max(100, std::atoi(argv[1]));
    if(argc>2) sim.steps = std::max(100, std::atoi(argv[2]));
    if(argc>3) sim.theta = std::max(0.3, std::atof(argv[3]));
    if(argc>4) sim.soft  = std::max(1e-6, std::atof(argv[4]));
    if(argc>5) sim.L     = std::max(1.0, std::atof(argv[5]));
    if(argc>6) sim.a_start = std::max(1e-4, std::atof(argv[6]));
    if(argc>7) sim.a_end   = std::max(sim.a_start+1e-4, std::atof(argv[7]));
    if(argc>8) sim.dump_every = std::max(1, std::atoi(argv[8]));

    // Cosmology (change here if desired)
    sim.cosmo.H0 = 1.0;          // code units; set a unit system later
    sim.cosmo.Omega_m = 0.315;   // Planck-ish
    sim.cosmo.Omega_L = 0.685;
    sim.cosmo.Omega_r = 0.0;     // late-time neglect

    std::cerr << "N="<<sim.N<<" steps="<<sim.steps<<" theta="<<sim.theta
              <<" soft="<<sim.soft<<" L="<<sim.L
              <<" a:["<<sim.a_start<<","<<sim.a_end<<"] dump_every="<<sim.dump_every<<"\n";

    sim.run();
    return 0;
}
