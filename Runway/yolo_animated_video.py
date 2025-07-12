import matplotlib.pyplot as plt
import cv2
import os

# --------- Step 1: Create simulation frames ---------
frames_dir = "sim_frames"
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

num_steps = 30

for i in range(num_steps):
    plt.figure(figsize=(6, 4))
    
    # Example: plot a dot moving
    plt.plot(i, 0, 'ro', markersize=20)
    plt.xlim(0, num_steps)
    plt.ylim(-1, 1)
    plt.title(f"Simulation Step {i+1}")
    plt.xlabel("Position")
    plt.ylabel("Fixed Height")
    plt.grid(True)
    
    # Save frame
    frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
    plt.savefig(frame_path)
    plt.close()

print("Simulation frames saved.")

# --------- Step 2: Create animation video ---------
images = sorted([os.path.join(frames_dir, img) for img in os.listdir(frames_dir) if img.endswith(".png")])
if images:
    # Read first image to get size
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    size = (width, height)

    out = cv2.VideoWriter('simulation_animation.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

    for img_path in images:
        img = cv2.imread(img_path)
        out.write(img)

    out.release()
    print("Animation video saved as simulation_animation.avi")

else:
    print("No frames found to create video.")
