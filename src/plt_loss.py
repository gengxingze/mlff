# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    dp0 = np.loadtxt(r'output\dp_sa.out', skiprows=0)
    dp1 = np.loadtxt(r'output\nn.out', skiprows=0)
    dp2 = np.loadtxt(r'output\mtp.out', skiprows=0)
    #plt.plot(dp0[:, 0], dp0[:, 2], label='dp_sa')
    plt.plot(dp0[:, 0], dp1[:, 2], label='nn')
    plt.plot(dp0[:, 0], dp2[:, 2], label='nn_embed')
    plt.axis([0, 50, 0, 0.01])
    plt.xlabel('epoch')
    plt.ylabel('energy')
    plt.grid(color='k', linestyle=':')
    plt.legend()
    plt.savefig('f2.png', dpi=128)
    plt.show()
    
    #plt.plot(dp0[:, 0], dp0[:, 3], label='dp_sa')
    plt.plot(dp0[:, 0], dp1[:, 3], label='nn')
    plt.plot(dp0[:, 0], dp2[:, 3], label='nn_embed')
    plt.axis([0, 50, 0, 0.1])
    plt.xlabel('epoch')
    plt.ylabel('force')
    plt.grid(color='k', linestyle=':')
    plt.legend()
    plt.savefig('f3.png', dpi=128)
    plt.show()



