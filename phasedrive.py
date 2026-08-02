import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. SIMULATION PARAMETERS
# ==========================================
# Grid and Drum Head
NRes = 300                # Grid resolution (N x N)
R = NRes/2 - 5               # Radius of the drum head
c = 0.2                # Speed of sound on the membrane
dx = 1.0               # Spatial step size
dt = 0.4               # Time step (must satisfy Courant condition: c*dt/dx < 0.707)
damping = 0.00005        # Slight damping to help standing waves stabilize

# Oscillator Parameters
A = 2.0                # Amplitude of the oscillator (mm)
wavelength = R / 5.87765331   # Enforce at least 5 wavelengths in the radius
freq = c / wavelength  # Driving frequency f
omega = 2 * np.pi * freq

# Movement Parameters
drift_speed = 2.35     # How fast the oscillator moves down the energy gradient

# ==========================================
# 2. INITIALIZATION
# ==========================================
# Create the 2D grid
x = np.arange(NRes)
y = np.arange(NRes)
X, Y = np.meshgrid(x, y)

# Circular drum head mask (True inside the drum, False outside)
drum_mask = (X - NRes/2)**2 + (Y - NRes/2)**2 <= R**2

# State matrices for the wave equation (current, previous, and next time steps)
u = np.zeros((NRes, NRes))
u_prev = np.zeros((NRes, NRes))
energy_envelope = np.zeros((NRes, NRes)) # Tracks time-averaged amplitude

# Start the oscillator slightly off-center so it has a gradient to follow
osc_x, osc_y = NRes/2 + 8.0, NRes/2 + 15.0 

# ==========================================
# 3. SETUP VISUALIZATION
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8))
fig.canvas.manager.set_window_title("Adaptive Drum Head Oscillator")

# Plot the wave amplitude
im = ax.imshow(u, cmap='RdBu', vmin=-A*1.5, vmax=A*1.5, origin='lower')

# Plot the boundary of the drum
circle = plt.Circle((NRes/2, NRes/2), R, color='black', fill=False, linewidth=3)
ax.add_patch(circle)

# Plot the oscillator's position
osc_marker, = ax.plot(osc_x, osc_y, 'ko', markersize=6, markerfacecolor='lime')

info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", 
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
ax.axis('off')

# ==========================================
# 4. MAIN FDTD LOOP
# ==========================================
time_step_counter = 0

def update(frame):
    global u, u_prev, energy_envelope, osc_x, osc_y, time_step_counter
    
    # Run multiple physics steps per visual frame to speed up the animation
    for _ in range(8):
        time_step_counter += 1
        t = time_step_counter * dt
        
        # 1. Calculate the spatial second derivatives (Laplacian) using slicing for speed
        laplacian = np.zeros((NRes, NRes))
        laplacian[1:-1, 1:-1] = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4*u[1:-1, 1:-1]) / dx**2
        
        # 2. Update the wave equation (Verlet integration with damping)
        u_next = (2 * u - u_prev * (1 - damping * dt) + (c * dt)**2 * laplacian) / (1 + damping * dt)
        
        # 3. Apply Boundary Conditions (clamped edges)
        u_next[~drum_mask] = 0.0
        
        # 4. Update the Energy Envelope (Low-pass filter of squared displacement)
        # This creates a map of the standing wave's intensity over time.
        energy_envelope = 0.99 * energy_envelope + 0.01 * (u**2)
        
        # 5. Move the Oscillator (Gradient Descent on Energy)
        # Calculate local gradient of the energy envelope around the oscillator
        ix, iy = int(osc_x), int(osc_y)
        if 2 < ix < NRes-2 and 2 < iy < NRes-2:
            osc_displacement = A * np.sin(omega * t)
            min_energy = float('inf')
            best_dx, best_dy = 0, 0
            
            for dy in [-1, 0, 1]:
                for dx_step in [-1, 0, 1]:
                    nx, ny = ix + dx_step, iy + dy
                    # Impedance proxy: squared difference between natural and forced displacement
                    energy_req = (u_next[ny, nx] - osc_displacement)**2
                    
                    if energy_req < min_energy:
                        min_energy = energy_req
                        best_dx = dx_step
                        best_dy = dy
                        
            # Move towards the point of minimum impedance
            if best_dx != 0 or best_dy != 0:
                norm = np.sqrt(best_dx**2 + best_dy**2)
                osc_x += drift_speed * (best_dx / norm) * dt
                osc_y += drift_speed * (best_dy / norm) * dt
            
            # Constrain to stay inside the drum
            dist_from_center = np.sqrt((osc_x - NRes/2)**2 + (osc_y - NRes/2)**2)
            if dist_from_center > R - 5:
                angle = np.arctan2(osc_y - NRes/2, osc_x - NRes/2)
                osc_x = NRes/2 + (R - 5) * np.cos(angle)
                osc_y = NRes/2 + (R - 5) * np.sin(angle)

        # 6. Apply Oscillator Forcing to the Membrane
        # We use a soft Gaussian footprint so the oscillator doesn't snap to integer grid lines
        footprint = np.exp(-((X - osc_x)**2 + (Y - osc_y)**2) / 2.0)
        osc_displacement = A * np.sin(omega * t)
        
        # Force the local displacement (soft blending based on footprint)
        u_next = u_next * (1 - footprint) + footprint * osc_displacement
        
        # 7. Advance time
        u_prev[:] = u
        u[:] = u_next

    # Update visual plots
    im.set_array(u)
    osc_marker.set_data([osc_x], [osc_y])
    info_text.set_text(f"Driven Drum Head\nf = {freq:.3f} Hz, X = {osc_x:.2f}, Y = {osc_y:.2f}")
    
    return im, osc_marker, info_text

# Run the animation
ani = animation.FuncAnimation(fig, update, frames=1000, interval=30, blit=True)
plt.show()