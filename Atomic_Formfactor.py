import numpy as np
import matplotlib.pyplot as plt

a = 1  # unit cell size
x = np.linspace(0, a, 1000)

# G · r for (111) and (200)
E_111 = np.cos(2 * np.pi * x / a)         # fundamental
E_200 = np.cos(4 * np.pi * x / a)         # second harmonic

plt.plot(x, E_111**2, label='|E(x)|² for (111)')
plt.plot(x, E_200**2, label='|E(x)|² for (200)', linestyle='--')

# mark atoms at 0 (A) and a/4 (B)
plt.axvline(0, color='k', linestyle=':', label='A site (0)')
plt.axvline(a/4, color='gray', linestyle=':', label='B site (a/4)')

plt.legend()
plt.xlabel('Position along unit cell')
plt.ylabel('Field intensity')
plt.title('Standing Wave Intensity Patterns')
plt.grid(True)
plt.show()
