import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

def plot_havelock_source_wake(U=5.0, nu=0.05):
    """
    Plots the wave height field for a single Havelock source.
    
    Parameters:
    U  : Ship speed (m/s)
    nu : Eddy viscosity coefficient (m^2/s)
    """
    # 1. Physical constants and setup
    g = 9.81
    k0 = g / U**2  # Fundamental wave number
    
    # 2. Define the spatial grid (x > 0 is behind the source)
    x = np.linspace(1, 50, 200) # Start slightly > 0 so eddy viscosity damping works
    y = np.linspace(-20, 20, 200)
    X, Y = np.meshgrid(x, y)
    
    # 3. Define the wave propagation angles (theta)
    # We avoid exactly +/- pi/2 to prevent division by zero before damping is applied
    theta = np.linspace(-np.pi/2 + 0.01, np.pi/2 - 0.01, 1000)
    
    # Reshape theta for vectorization: (len(theta), 1, 1) to broadcast against (Y, X)
    theta_3d = theta[:, np.newaxis, np.newaxis]
    sec_theta = 1.0 / np.cos(theta_3d)
    
    # 4. Construct the integrand
    # Phase component: k0 * sec^2(theta) * (x*cos(theta) + y*sin(theta))
    phase = k0 * sec_theta**2 * (X * np.cos(theta_3d) + Y * np.sin(theta_3d))
    
    # Amplitude component: sec^4(theta). 
    # For a point source, the ship hull integral (P + iQ) simplifies to a constant (e.g., 1.0).
    amplitude = sec_theta**4 
    
    # Eddy viscosity damping factor to remove unphysical ripples and bound the infinity
    damping = np.exp(-2 * nu * k0**2 * (X / U) * sec_theta**4)
    
    # Combine to form the complex integrand
    integrand = np.exp(-1j * phase) * amplitude * damping
    
    # 5. Integrate over theta using Simpson's rule and extract the imaginary part
    # Formula: Z_F(x,y) = -(2 * k0^2 / pi) * Im { Integral( ... ) }
    integral_result = np.imag(simpson(integrand, x=theta, axis=0))
    Z_F = - (2 * k0**2 / np.pi) * integral_result
    
    # 6. Plotting the wave height field
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X, Y, Z_F, levels=50, cmap='RdBu_r')
    plt.colorbar(contour, label='Wave Height $Z_F$')
    plt.title(f'Far-Field Wave Pattern of a Havelock Source ($U={U}$ m/s)')
    plt.xlabel('Distance behind source, x (m)')
    plt.ylabel('Transverse distance, y (m)')
    plt.show()

if __name__ == "__main__":
    plot_havelock_source_wake(U=5.0)