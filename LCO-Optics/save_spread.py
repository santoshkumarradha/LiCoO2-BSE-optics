import numpy as np
import re
from scipy.interpolate import Rbf
from scipy.interpolate import Rbf
from scipy.stats import entropy
def save_wave(nk1=6,nk2=6,nk3=6,v=10,c=5,n=200):
    def extract_nums(text):
        numeric_const_pattern = r'[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?:[Ee] [+-]? \d+ ) ?'
        rx = re.compile(numeric_const_pattern, re.VERBOSE)
        num=[float(i) for i in rx.findall(text)]
        try:
            return num[0]+1j*num[1]
        except:None
    def get_eig_fromquestaal(eig_fname):
        eigenvecs=[]
        cnt=0
        with open(eig_fname) as topo_file:
                for line in topo_file:
                    #print(cnt)
                    if cnt==nk1*nk2*nk3*v*c*n:
                        break
                    if "val" not in line:
                        eigenvecs.append(extract_nums(line))
                        cnt+=1
        return eigenvecs

    vals=get_eig_fromquestaal("Eigenvecs_bse")
    vals=np.array(vals).reshape(-1,nk1*nk2*nk3*v*c)

    np.save("wave_data.npy",vals)
    return vals
sq = lambda a: np.real(a*a.conj()) 
N=4000    
wave=save_wave(nk1=6,nk2=6,nk3=6,v=10,c=5,n=N)
wave=wave.reshape(-1,6**3,10,5)
print("Done saving, now calculating .....")

kpts=np.load("kpts.npy")
kpts_calc=np.load("kpts_calc.npy")
def get_entropy(n):
    rbfi = Rbf(kpts.T[0], kpts.T[1], kpts.T[2],sq(wave[n]).sum(axis=1).sum(axis=1) ,smooth=.00)
    w=rbfi(kpts_calc.T[0], kpts_calc.T[1], kpts_calc.T[2])
    w[w<0]=0
    w/=w.sum()
    return 1-entropy(w)/np.log(len(w))
np.save("spread.npy",[get_entropy(i) for i in range(N)])