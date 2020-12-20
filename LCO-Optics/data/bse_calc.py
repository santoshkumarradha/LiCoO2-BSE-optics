import subprocess
import functools
import time


def calc(cond, val):
    print(f"calculating {val} {cond}")
    subprocess.call(
        [f"sed 's/NumValStates.*/NumValStates {val}/' gwinp>GWinput"],
        shell=True)
    subprocess.call(
        [f"sed -i 's/NumCondStates.*/NumCondStates {cond}/' GWinput"],
        shell=True)
    subprocess.call([
        f"echo 0 | /home/srr70/codes/bse/build/code2/bethesalpeter > BSE_out_val_{val}_cond_{cond}"
    ],
                    shell=True)
    subprocess.call([f"cp eps_BSE.out BSE_val_{val}_cond_{cond}"], shell=True)
    subprocess.call([f"cp Eigenvals_bse Eigenvals_val_{val}_cond_{cond}"],
                    shell=True)


for val in [1, 2, 3, 4, 5]:
    for cond in [1, 2, 3, 4, 5]:
        start = time.time()
        print("started calculating at {}".format(
            time.strftime("%H:%M:%S", time.gmtime(start))))
        calc(cond, val)
        total = time.time() - start
        print("total time = {}".format(
            time.strftime("%H:%M:%S", time.gmtime(total))))
