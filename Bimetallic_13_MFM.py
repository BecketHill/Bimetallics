import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar


# Defining compounds and assigning material
# All from Herbst, Croat [1982].

dy = {"name":"Dy",
    "rho":8.84, "A":330, "theta":0, "nff":7660, "nrf":-1060, "nrr":408,
    "Jr":15/2, "Sr":5/2, "Lr":5, "Jf":1, "Sf":5/2, "Lf":0, "muf0":1.9
}

er = {"name":"Er",
    "rho":9.08, "A":335, "theta":0, "nff":7760, "nrf":-842, "nrr":96,
    "Jr":15/2, "Sr":3/2, "Lr":6, "Jf":1, "Sf":5/2, "Lf":0, "muf0":2.5
}

tm = {"name":"Tm",
    "rho":9.78, "A":337, "theta":0, "nff":7000, "nrf":-761, "nrr":161,
    "Jr":6, "Sr":1, "Lr":5, "Jf":6, "Sf":5/2, "Lf":0, "muf0":2.3
}

mat = er

# ---------------------------
# Input parameters
# ---------------------------
dT = 1.0
Tm = 700  # maximum temperature index

# physical constants
NA = 6.02e23       # Avogadro
muB = 9.274e-21    # Bohr magneton, erg/G
kB = 1.381e-16     # Boltzmann, erg/K
rho = mat["rho"]   #density DyFe3, g/cc
A = mat["A"]       #atomic weight DyFe3, g/mol

# applied field, Gauss
#h = 1.6*10**4
h = 0

# field coefficients, dimensionless
# canting model

nff,nrf,nrr = mat["nff"], mat["nrf"], mat["nrr"]

theta = mat["theta"] #degrees canting, Herbst
gamma = np.cos(theta*np.pi/180)

# angular momenta

Jr, Sr, Lr = mat["Jr"] * gamma, mat["Sr"], mat["Lr"]
Jf, Sf, Lf = mat["Jf"], mat["Jr"], mat["Jr"]

# conversion factor, erg/(Gauss cc) = Gauss

d = NA*muB*rho/A

# ---------------------------
# g-factors
# ---------------------------
gr = 1 + (Jr * (Jr + 1) + Sr * (Sr + 1) - Lr * (Lr + 1)) / (2 * Jr * (Jr + 1))
grs = 1 + (Sr * (Sr + 1) - Lr * (Lr + 1)) / (Jr * (Jr + 1))
gf = 1 + (Jf * (Jf + 1) + Sf * (Sf + 1) - Lf * (Lf + 1)) / (2 * Jf * (Jf + 1))

gr = 4/3 #From Herbst
gf = 2.0

# ---------------------------
# Initial magnetization, T = 0
# ---------------------------
mur0 = gr * Jr
muf0 = mat["muf0"]



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
# Mathematica used muc[n], mua[n], mud[n]
# Here we store them as numpy arrays
# ---------------------------
mur = np.zeros(Tm + 1)
muf = np.zeros(Tm + 1)

# Initial conditions
mur[0] = mur0
muf[0] = -muf0

# ---------------------------
# Solve recurrence equations
# ---------------------------

N = Tm

for n in range(N):
    T = dT * (n + 1)
    # molecular fields
    hr = h + d*(1*nrr*mur[n] + 3*nrf*muf[n]);
    hf = h + d*(3*nff*muf[n] + 1*nrf*mur[n]);

    xr = mur0 * muB * hr / (kB * T)
    xf = muf0 * muB * hf / (kB * T)

    # Brillouin functions
    bjr = brillouin(Jr, np.array([xr]))[0]
    bjd = brillouin(Jf, np.array([xf]))[0]

    # recurrence
    mur[n + 1] = mur0 * bjr
    muf[n + 1] = muf0 * bjd

# ---------------------------
# Build T-dependent lists
# ---------------------------
Tvals = np.arange(1, N + 1) * dT

murt = np.column_stack([Tvals, 1 * mur[1:]])
muft = np.column_stack([Tvals, 3 * -muf[1:]])
must = np.column_stack([Tvals, 1 * mur[1:] + 3 * muf[1:]])

# ---------------------------
# Interpolation
# ---------------------------
imurt = interp1d(murt[:, 0], murt[:, 1], kind="linear", fill_value="extrapolate")
imuft = interp1d(muft[:, 0], muft[:, 1], kind="linear", fill_value="extrapolate")
imust = interp1d(must[:, 0], must[:, 1], kind="linear", fill_value="extrapolate")
#print(must)

# ---------------------------
# Find Tcomp
# ---------------------------
sol = root_scalar(lambda T: float(imust(T)), bracket=[10, 400], method="brentq")
tc = sol.root

# ---------------------------
# Contributions at Tcomp
# ---------------------------
murtc = float(imurt(tc))
muftc = float(imuft(tc))
srtc = (grs / gr) * murtc
sxtc = srtc + muftc

# ---------------------------
# Spin excess
# ---------------------------
def sx(T):
    return (grs / gr) * float(imurt(T)) - float(imuft(T))

sxt = np.array([sx(T) for T in range(2, Tm+1)])
imurtst = np.column_stack([np.arange(2, Tm+1), [(grs / gr) * float(imurt(T)) for T in range(2, Tm+1)]])
imurtot = np.column_stack([np.arange(2, Tm+1), [(1 - grs / gr) * float(imurt(T)) for T in range(2, Tm+1)]])

# ---------------------------
# Prints
# ---------------------------
#print(f"gamma                          = {gamma}")
#print(f"Jr                             = {Jr}")
#print(f"gc                             = {gc}")
#print(f"gcs                            = {gcs}")
print(f"Tcomp (K)                         = {tc}")
#print(f"contribution of Tb at Tc (uB)  = {muctc}")
#print(f"spin component at Tc (uB)      = {Srtc}")
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

Tplot = np.linspace(1, Tm, 1000)
sxplot = np.array([abs(sx(T)) for T in Tplot])

plt.figure(figsize=(8, 5))
plt.title(f"Moment and Spin Model for {mat["name"]}Fe3")
plt.plot(murt[:, 0], murt[:, 1], label=f"{mat["name"]}")
plt.plot(muft[:, 0], muft[:, 1], label="Fe3")
plt.plot(must[:, 0], must[:, 1], label="Total Moment")
plt.plot(imurtst[:, 0], imurtst[:, 1], "--", label=f"{mat["name"]} Spin")
plt.plot(imurtot[:, 0], imurtot[:, 1], "--", label=f"{mat["name"]} Orbital")
plt.plot(Tplot, sxplot, label="Total Spin Excess")
plt.xlabel("T (K)")
plt.ylabel("M (uB/molecule)")
plt.grid(True)
plt.legend()
#plt.tight_layout()
plt.show()



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
