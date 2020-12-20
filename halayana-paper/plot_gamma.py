import matplotlib.pyplot as plt
import numpy as np


def get_qpts_from_file(fname="out"):
    with open(fname, 'r') as fhand:
        lines = [line[:-1] for line in fhand if line.strip() != '']
    qx = []
    qy = []
    qz = []
    index = []
    for i in lines:
        split = i.split()
        if len(split) == 7 and split[0][0] == "(":
            qx.append(float(split[1]))
            qy.append(float(split[2]))
            qz.append(float(split[3]))
            index.append(int(split[4]))
    return np.array(qx), np.array(qy), np.array(qz), np.array(index)


def get_data(fname="popt.temp"):
    print("Reading file\n")
    with open('popt.temp', 'r') as fhand:
        lines = [line[:-1] for line in fhand if line.strip() != '']

    print("extracting numbers\n")
    a = []
    for i in lines[1:]:
        a.append(list(map(float, filter(None, i.split(" ")))))
    np.save("opt_data", a)
    print("Done\n")
    #return np.array(a)


def plot_k(omega=6, proj=[0, 1], eps_axis=0, ax=None, get_from_file=True):
    '''omega: energy value
    proj: projection to plot 0-x 1-y 2-z
    eps_axis: orientation of polarization 0-x 1-y 2-z
    ax: axis object'''
    if get_from_file:
        qx, qy, qz, index = get_qpts_from_file(fname="out")
        get_data(fname="popt.temp")
        data = np.load("opt_data.npy")
        data = data.reshape(2000, -1)
    else:
        data = np.load("data/full_data.npy")
        data = data.reshape(2000, -1)
        qpts_data = np.loadtxt("data/qpts", delimiter=",", skiprows=2)
        qx = qpts_data.T[0]
        qy = qpts_data.T[1]
        qz = qpts_data.T[2]
        index = np.array(qpts_data.T[3] - 1, dtype=np.int)

    def find_nearest(value, array=data.T[0] * 13.6):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or np.fabs(value - array[idx - 1]) <
                        np.fabs(value - array[idx])):
            return idx - 1
        else:
            return idx

    indx = find_nearest(omega)
    eps = []
    x = []
    y = []
    z = []
    if ax == None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.grid("on", linewidth=0.5)
    if eps_axis < 3:
        for i in index:
            x.append(qx[i])
            y.append(qy[i])
            z.append(qz[i])
            eps.append(data.T[1:].reshape(3, -1, 2000)[eps_axis, :, indx][i])
    else:
        for i in index:
            x.append(qx[i])
            y.append(qy[i])
            z.append(qz[i])
            tmp=data.T[1:].reshape(3,-1,2000)[0,:,0][i]+\
                data.T[1:].reshape(3,-1,2000)[1,:,0][i]+\
                data.T[1:].reshape(3,-1,2000)[2,:,0][i]
            eps.append(tmp)
    eps /= np.linalg.norm(eps)

    r = ax.tricontourf(np.vstack((x, y, z))[proj[0]],
                       np.vstack((x, y, z))[proj[1]],
                       eps,
                       50,
                       cmap="RdYlBu")
    ax.scatter(np.vstack((x, y, z))[proj[0]],
               np.vstack((x, y, z))[proj[1]],
               s=15,
               c="w",
               edgecolor="k",
               alpha=.09)
    #     ax.tricontour(np.vstack((x,y,z))[proj[0]],np.vstack((x,y,z))[proj[1]],eps);
    plt.colorbar(r, ax=ax)
    ax.set_xlabel("k$_i$")
    ax.set_ylabel("k$_j$")


fig, ax1 = plt.subplots(1, 2, figsize=(12, 5))

omega = 5.9

ax = ax1[1]
plot_k(omega=omega, proj=[0, 1], ax=ax, eps_axis=2)
ax.grid("on")

ax = ax1[0]
data = np.load("data/full_data.npy")
data = data.reshape(2000, -1)
y = data.T[1:].reshape(3, 146, 2000)  # axis,kpts,omega
dos = pydos(data.T[0] * 13.6, y[0:3, 0, :].sum(axis=0))
dos.smear(9e-2)
dos.normalize(mode='max', value=1)
c = "#e63946"
ax.plot(dos.x, dos.y, c=c, label="$<\\Gamma>$")
ax.fill_between(dos.x, dos.y, color=c, alpha=0.1)
ax.set_ylabel("Im$[\\epsilon]$")
ax.set_xlabel("Energy (eV)")

ax.axvline(omega, ls="--", lw=2, c="k")
ax.legend()
plt.tight_layout()
plt.savefig("halyana_near_gamma.png", dpi=300)
plt.show()