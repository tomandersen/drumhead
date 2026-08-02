import signal
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. SIMULATION PARAMETERS
# ==========================================
# Grid and Drum Head
NRes = 400                # Grid resolution (N x N)
R = NRes/2 - 5               # Radius of the drum head
c = 0.2                # Speed of sound on the membrane
dx = 1.0               # Spatial step size
dt = 0.2               # Time step (must satisfy Courant condition: c*dt/dx < 0.707)
damping = 0.00005        # Slight damping to help standing waves stabilize

# Oscillator Parameters
A = 2.0                # Amplitude of the oscillator (mm)
wavelength = R / 5.87765331   # Enforce at least 5 wavelengths in the radius
freq = c / wavelength  # Driving frequency f
omega = 2 * np.pi * freq

# Movement Parameters
drift_speed = 4     # How fast the oscillator moves down the energy gradient

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
osc_x, osc_y = NRes/2 + 8.0, NRes/2 + NRes/4 

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
        osc_displacement = A * np.sin(omega * t)
        osc_velocity = A * omega * np.cos(omega * t)
        if 2 < ix < NRes-2 and 2 < iy < NRes-2:
            # Bjerknes Force:
            # the gradient of the pressure u_next combined with the amplitude of the driver
            # gives us a force...
            
            # Helper to get gradient at a specific integer grid point
            def get_grad(cx, cy):
                gx = (u_next[cy, cx+1] - u_next[cy, cx-1]) / (2 * dx)
                gy = (u_next[cy+1, cx] - u_next[cy-1, cx]) / (2 * dx)
                return gx, gy
                
            gx00, gy00 = get_grad(ix, iy)
            gx10, gy10 = get_grad(ix+1, iy)
            gx01, gy01 = get_grad(ix, iy+1)
            gx11, gy11 = get_grad(ix+1, iy+1)
            
            # Bilinearly interpolate the gradient to the exact sub-pixel oscillator position
            fx = osc_x - ix
            fy = osc_y - iy
            
            grad_x = (gx00 * (1-fx)*(1-fy) + gx10 * fx*(1-fy) + 
                      gx01 * (1-fx)*fy + gx11 * fx*fy)
            grad_y = (gy00 * (1-fx)*(1-fy) + gy10 * fx*(1-fy) + 
                      gy01 * (1-fx)*fy + gy11 * fx*fy)
                      
            # The instantaneous Bjerknes force is proportional to the gradient of the field 
            # times the instantaneous displacement (amplitude) of the driver.
            force_x = -osc_displacement * grad_x
            force_y = -osc_displacement * grad_y
            
            # Move the oscillator
            osc_x += drift_speed * force_x * dt
            osc_y += drift_speed * force_y * dt
            
            # Constrain to stay inside the drum
            dist_from_center = np.sqrt((osc_x - NRes/2)**2 + (osc_y - NRes/2)**2)
            if dist_from_center > R - 5:
                angle = np.arctan2(osc_y - NRes/2, osc_x - NRes/2)
                osc_x = NRes/2 + (R - 5) * np.cos(angle)
                osc_y = NRes/2 + (R - 5) * np.sin(angle)

        # 6. Apply Oscillator Forcing to the Membrane
        # We use a soft Gaussian footprint so the oscillator doesn't snap to integer grid lines
        # We only need to go a few grid points away from the oscillation center, 
        # no need to use the whole grid.  Radius of influence is 5 grid points.
        
        footprint = np.exp(-((X - osc_x)**2 + (Y - osc_y)**2) / 2.0)
        
        # Force the local displacement (soft blending based on footprint)
        # The oscillator will drive the wave, but no higher than its osc_displacement,
        # so u_next will be at most osc_displacement times the footprint (which is 1 at the center).
        # The wave can have higher amplitudes away from the oscillator if energy builds up there.
        # amplitude of the oscilation added to the wave is zero if the u_max amplitude
        # is already bigger than the osc_displacement.
        # or perhaps an energy model. The particle can absorb or emit energy into the grid, 
        # energy is conserved. It emits energy if the wave is pushing it, abosorbs when phase is matched
        
        # what is u_next at the location of the oscilator, properly interpolated?
        ix, iy = int(osc_x), int(osc_y)
        fx = osc_x - ix
        fy = osc_y - iy
        
        u_osc = u_next[iy, ix] * (1-fx)*(1-fy) + u_next[iy, ix+1] * fx*(1-fy) + \
                u_next[iy+1, ix] * (1-fx)*fy + u_next[iy+1, ix+1] * fx*fy
        
        
        # now add some noise to the amplitude to help it escape local minima
        # the random kick should be added to the oscilator amplitude, not the wave amplitude
        # in addition the random noise needs to be at all frequencies, not just high frequency as it is now.
        random_kick = 2 *A* random.uniform(-1,1)
        osc_displacement = osc_displacement + random_kick
        

        
        # Is the oscillator in phase with the wave or out of phase?
        # 
        if u_osc * osc_displacement <= 0:
            # we are out of phase, so we want to push the wave in the same direction as the oscillator
            field_addition = osc_displacement
        else:
            # we are in phase, so we want to push the wave in the same direction as the oscillator
            diff = osc_displacement - u_osc
            if osc_displacement > 0:
                field_addition = max(diff, 0)
            else:
                field_addition = min(diff, 0)

        u_next = u_next * (1 - footprint) + footprint * field_addition
        
        # 7. Advance time
        u_prev[:] = u
        u[:] = u_next

    # Update visual plots
    im.set_array(u)
    osc_marker.set_data([osc_x], [osc_y])
    info_text.set_text(f"Driven Drum Head\nf = {freq:.3f} Hz, X = {osc_x:.5f}, Y = {osc_y:.5f}")
    
    return im, osc_marker, info_text

# Run the animation
# Comment out the animation for debugging:
ani = animation.FuncAnimation(fig, update, frames=1000, interval=30, blit=True)
plt.show()

# Add this to step through cleanly in the debugger:
# if __name__ == "__main__":
#     for i in range(5):
#         update(i)
