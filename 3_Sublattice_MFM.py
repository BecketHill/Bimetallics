import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

#Make it take user input on which rare earth it's using!


tb = {"name":"Tb",
    "rho":8.5, "A":2238, "gamma":0.32, "nac":-4.2, "ncc":0.0, "ncd":6.5,
    "Jc":6, "Sc":3, "Lc":3, "gc":3/2, "Jf":1, "Sa":5/2, "La":0
}

dy = {"name":"Dy",
    "rho":8.57, "A":2259, "gamma":0.41, "nac":-4, "ncc":0.0, "ncd":6,
    "Jc":7.5, "Sc":2.5, "Lc":5, "gc":5/4, "Jf":1, "Sa":5/2, "La":0
}

ho = {"name":"Ho",
    "rho":8.67, "A":2274, "gamma":0.37, "nac":-2.1, "ncc":0.0, "ncd":4,
    "Jc":8, "Sc":2, "Lc":6, "gc":5/4, "Jf":1, "Sa":5/2, "La":0
}

er = {"name":"Er",
    "rho":8.79, "A":2288, "gamma":0.42, "nac":-0.2, "ncc":0.0, "ncd":2.2,
    "Jc":7.5, "Sc":1.5, "Lc":6, "gc":6/5, "Jf":1, "Sa":5/2, "La":0
}

tm = {"name":"Tm",
    "rho":8.89, "A":2298, "gamma":0.3, "nac":-0.36, "ncc":0.0, "ncd":6.1,
    "Jc":6, "Sc":1, "Lc":5, "gc":7/6, "Jf":1, "Sa":5/2, "La":0
}

yb = {"name":"Yb",
    "rho":8.9, "A":2323, "gamma":0.33, "nac":-3.4, "ncc":0.0, "ncd":6.8,
    "Jc":3.5, "Sc":0.5, "Lc":3, "gc":8/7, "Jf":1, "Sa":5/2, "La":0
}

mat = dy

# ---------------------------
# Input parameters
# ---------------------------
dT = 1.0
Tm = 400  # maximum temperature index

# physical constants
NA = 6.02e23       # Avogadro
muB = 9.274e-21    # Bohr magneton, erg/G
kB = 1.381e-16     # Boltzmann, erg/K

# field coefficients in mol/cc
naa = -65
nad = 97
ndd = -30.4

nac = mat["nac"]
ncc = mat["ncc"]
ncd = mat["ncd"]

# angular momenta
gamma = mat["gamma"]
Sc = mat["Sc"]
Lc = gamma * mat["Lc"]
Jc = Lc + Sc

Sa = mat["Sa"]
Sd = mat["Sa"]

# ---------------------------
# g-factors
# ---------------------------
gc = 1 + (Jc * (Jc + 1) + Sc * (Sc + 1) - Lc * (Lc + 1)) / (2 * Jc * (Jc + 1))
gcs = 1 + (Sc * (Sc + 1) - Lc * (Lc + 1)) / (Jc * (Jc + 1))
ga = 2.0
gd = 2.0

# ---------------------------
# Initial magnetization, T = 0
# ---------------------------
muc0 = 3.0 * gc * Jc * muB * NA
mua0 = 2.0 * ga * Sa * muB * NA
mud0 = 3.0 * gd * Sd * muB * NA

# ---------------------------
# Helper functions
# ---------------------------
def coth(x):
    """Numerically stable coth."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    small = np.abs(x) < 1e-10
    out[~small] = 1.0 / np.tanh(x[~small])

    # For tiny x, coth(x) ~ 1/x + x/3, but here just avoid divide-by-zero.
    # Since the Brillouin combination cancels the singularity, using a tiny-x expansion
    # for B_J is better than relying on coth directly.
    out[small] = 1.0 / x[small]  # not usually used directly for tiny x in final code
    return out


def brillouin(J, x):
    """
    Brillouin function:
    B_J(x) = ((2J+1)/(2J)) coth((2J+1)x/(2J)) - (1/(2J)) coth(x/(2J))
    """
    x = np.asarray(x, dtype=float)
    BJ = np.empty_like(x)

    small = np.abs(x) < 1e-8
    large = ~small

    # small-x expansion: B_J(x) ~ ((J+1)/(3J)) x
    BJ[small] = ((J + 1.0) / (3.0 * J)) * x[small]

    a = (2.0 * J + 1.0) / (2.0 * J)
    BJ[large] = a * (1.0 / np.tanh(a * x[large])) - (1.0 / (2.0 * J)) * (1.0 / np.tanh(x[large] / (2.0 * J)))

    return BJ


# ---------------------------
# Recurrence arrays
# ---------------------------
muc = np.zeros(Tm + 1)
mua = np.zeros(Tm + 1)
mud = np.zeros(Tm + 1)

# Initial conditions
muc[0] = muc0
mua[0] = mua0
mud[0] = mud0

# ---------------------------
# Solve recurrence equations
# ---------------------------
for n in range(Tm):
    # molecular fields
    xc = (Jc * gc * muB) / (kB * dT * (n + 1)) * (ncc * muc[n] + nac * mua[n] + ncd * mud[n])
    xa = (Sa * ga * muB) / (kB * dT * (n + 1)) * (naa * mua[n] + nac * muc[n] + nad * mud[n])
    xd = (Sd * gd * muB) / (kB * dT * (n + 1)) * (ndd * mud[n] + ncd * muc[n] + nad * mua[n])

    # Brillouin functions
    bjc = brillouin(Jc, np.array([xc]))[0]
    bja = brillouin(Sa, np.array([xa]))[0]
    bjd = brillouin(Sd, np.array([xd]))[0]

    # recurrence
    muc[n + 1] = muc0 * bjc
    mua[n + 1] = mua0 * bja
    mud[n + 1] = mud0 * bjd

# ---------------------------
# Build T-dependent lists
# ---------------------------
Tvals = np.arange(1, Tm + 1) * dT

muct = np.column_stack([Tvals, muc[1:] / (muB * NA)])
muat = np.column_stack([Tvals, mua[1:] / (muB * NA)])
mudt = np.column_stack([Tvals, -mud[1:] / (muB * NA)])
must = np.column_stack([Tvals, (muc[1:] + mua[1:] - mud[1:]) / (muB * NA)])

# ---------------------------
# Interpolation
# ---------------------------
imuct = interp1d(muct[:, 0], muct[:, 1], kind="linear", fill_value="extrapolate")
imuat = interp1d(muat[:, 0], muat[:, 1], kind="linear", fill_value="extrapolate")
imudt = interp1d(mudt[:, 0], mudt[:, 1], kind="linear", fill_value="extrapolate")
imust = interp1d(must[:, 0], must[:, 1], kind="linear", fill_value="extrapolate")

# ---------------------------
# Find Tc
# Mathematica:
# tc = FindRoot[imust[T] == 0., {T, 100.}][[1,2]]
# ---------------------------
sol = root_scalar(lambda T: float(imust(T)), bracket=[10, 300.0], method="brentq")
tc = sol.root

# ---------------------------
# Contributions at Tc
# ---------------------------
muctc = float(imuct(tc))
muatc = float(imuat(tc))
mudtc = float(imudt(tc))
sctc = (gcs / gc) * muctc
sxtc = sctc + muatc + mudtc

# ---------------------------
# Spin excess
# ---------------------------
def sx(T):
    return (gcs / gc) * float(imuct(T)) + float(imuat(T)) + float(imudt(T))

sxt = np.array([sx(T) for T in range(2, 401)])
imuctst = np.column_stack([np.arange(2, 401), [(gcs / gc) * float(imuct(T)) for T in range(2, 401)]])
imuctot = np.column_stack([np.arange(2, 401), [(1 - gcs / gc) * float(imuct(T)) for T in range(2, 401)]])

# ---------------------------
# Prints
# ---------------------------
#print(f"gamma                          = {gamma}")
#print(f"Jc                             = {Jc}")
#print(f"gc                             = {gc}")
#print(f"gcs                            = {gcs}")
print(f"Tc (K)                         = {tc}")
#print(f"contribution of Tb at Tc (uB)  = {muctc}")
#print(f"spin component at Tc (uB)      = {sctc}")
#print(f"contribution of 2Fe at Tc (uB) = {muatc}")
#print(f"contribution of 3Fe at Tc (uB) = {mudtc}")
#print(f"spin excess at Tc (uB)         = {sxtc}")

# moments at particular temperatures
Treq = [tc - 10.0, tc, tc + 10.0, 296.0]
Mreq = [float(imust(T)) for T in Treq]

#for T, M in zip(Treq, Mreq):
    #print(f"M at {T:.3f} K = {M:.6f} uB/molecule")

# ---------------------------
# Plots
# ---------------------------

plt.figure(figsize=(8, 5))
plt.plot(muct[:, 0], muct[:, 1], label=f"3{mat["name"]} (c)")
plt.plot(muat[:, 0], muat[:, 1], label=f"2Fe (a)")
plt.plot(mudt[:, 0], mudt[:, 1], label=f"3Fe (d)")
plt.plot(must[:, 0], must[:, 1], label="Total")
plt.plot(imuctst[:, 0], imuctst[:, 1], "--", label=f"3{mat["name"]} Spin")
plt.plot(imuctot[:, 0], imuctot[:, 1], "--", label=f"3{mat["name"]} Orbital")
plt.xlabel("T (K)")
plt.ylabel("M (uB/molecule)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

Tplot = np.linspace(1, 300, 1000)
sxplot = np.array([sx(T) for T in Tplot])

#plt.figure(figsize=(8, 5))
#plt.plot(Tplot, sxplot, label="Spin Density")
#plt.axvline(tc, linestyle="--")
#plt.text(tc + 12, 0.2, "T = Tc")
#plt.xlabel("T (K)")
#plt.ylabel("Spin Density (uB/molecule)")
#plt.xlim(100, 300)
#plt.ylim(-1, 5)
#plt.grid(True)
#plt.tight_layout()
#plt.show()

T_spr8 = np.array([300, 288, 270, 245, 228, 100])
sx_vals = np.array([sx(T+8) for T in T_spr8])
#print(sx_vals)
print("Spin excess at Tcomp: ", sx(tc))
