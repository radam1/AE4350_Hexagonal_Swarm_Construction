import matplotlib.pyplot as plt
import numpy as np

# Plot 1: Stability Analysis for the Same Setup
p1_struts_created = np.array([480, 482, 469, 479, 469, 470, 477, 463, 489, 475])  
p1_std = np.std(p1_struts_created)
p1_min = np.min(p1_struts_created)
p1_max = np.max(p1_struts_created)
p1_mean = np.mean(p1_struts_created)

print("For Stability Analysis:\n" \
    f"Max Struts: {p1_max}\n" \
    f"Min Struts: {p1_min}\n" \
    f"Std. Of Struts: {p1_std}\n" \
    f"Mean # of Struts: {p1_mean}")

# Plot 2: Number of Agents vs Number of Struts
num_agents = np.array([1, 3, 5, 7, 10, 20])
p2_struts_created = np.array([192, 348, 480, 591, 673, 881])
p2_radius = np.array([131-72.5, 142-54, 149-51, 152-48, 157-43, 163-35])

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
#fig.suptitle('Sensitivity to', fontsize=24)

axes[0].plot(num_agents, p2_struts_created, 's-')
axes[0].set_title('Struts vs #Agents', fontsize=24)
axes[0].set_xlabel('Number of Agents', fontsize=24)
axes[0].set_ylabel('Number of Struts', fontsize=24)
axes[0].tick_params(axis='both', labelsize=20)
axes[0].set_ylim([0, 1200])
axes[0].grid(True)

axes[1].plot(num_agents, p2_radius, 's-')
axes[1].set_title('Radius vs #Agents', fontsize=24)
axes[1].set_xlabel('Number of Agents', fontsize=24)
axes[1].set_ylabel('Radius of Structure [mm]', fontsize=24)
axes[1].tick_params(axis='both', labelsize=20)
axes[1].set_ylim([0, 200])
axes[1].grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('num_agents_analysis.pdf')


# Plot 3: Number of Steps vs Number of Struts
num_steps = np.array([5000, 10000, 25000, 50000])
p3_struts_created = np.array([101, 169, 309, 480])
p3_radius = np.array([120-75, 127.5-72.5, 140-62.5, 149-51])

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
#fig.suptitle('Steps vs Struts & Radius', fontsize=24)

axes[0].plot(num_steps, p3_struts_created, 's-')
axes[0].set_title('Struts vs #Steps', fontsize=24)
axes[0].set_xlabel('Number of Steps', fontsize=24)
axes[0].set_ylabel('Number of Struts', fontsize=24)
axes[0].tick_params(axis='both', labelsize=20)
axes[0].set_ylim([0, 1200])
axes[0].grid(True)

axes[1].plot(num_steps, p3_radius, 's-')
axes[1].set_title('Radius vs #Steps', fontsize=24)
axes[1].set_xlabel('Number of Steps', fontsize=24)
axes[1].set_ylabel('Radius', fontsize=24)
axes[1].tick_params(axis='both', labelsize=20)
axes[1].set_ylim([0, 200])
axes[1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('NumSteps_analysis.pdf')

# Plot 4: Strut Length vs Struts & Radius
strut_length = np.array([1, 4, 7, 10])
p4_struts_created = np.array([996, 480, 339, 273])
p4_radius = np.array([118-83, 149-51, 174-26, 195-10])

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
#fig.suptitle('Strut Length vs Struts & Radius', fontsize=24)

axes[0].plot(strut_length, p4_struts_created, 's-')
axes[0].set_title('Struts vs Length', fontsize=24)
axes[0].set_xlabel('Strut Length', fontsize=24)
axes[0].set_ylabel('Number of Struts', fontsize=24)
axes[0].tick_params(axis='both', labelsize=20)
axes[0].set_ylim([0, 1200])
axes[0].grid(True)

axes[1].plot(strut_length, p4_radius, 's-')
axes[1].set_title('Radius vs Length', fontsize=24)
axes[1].set_xlabel('Strut Length', fontsize=24)
axes[1].set_ylabel('Radius', fontsize=24)
axes[1].tick_params(axis='both', labelsize=20)
axes[1].set_ylim([0, 200])
axes[1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('TargetLength_analysis.pdf')

# Plot 5: Length Error vs Struts & Radius
errors = np.array([0, 0.5, 1, 2])
p5_struts_created = np.array([480, 499, 755, 1095])
p5_radius = np.array([149-51, 140-54, 135-63, 125-75])

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
#fig.suptitle('Length Error vs Struts & Radius', fontsize=24)

axes[0].plot(errors, p5_struts_created, 's-')
axes[0].set_title('Struts vs Error', fontsize=24)
axes[0].set_xlabel('Error Magnitude', fontsize=24)
axes[0].set_ylabel('Number of Struts', fontsize=24)
axes[0].tick_params(axis='both', labelsize=20)
axes[0].set_ylim([0, 1200])
axes[0].grid(True)

axes[1].plot(errors, p5_radius, 's-')
axes[1].set_title('Radius vs Error', fontsize=24)
axes[1].set_xlabel('Error Magnitude', fontsize=24)
axes[1].set_ylabel('Radius', fontsize=24)
axes[1].tick_params(axis='both', labelsize=20)
axes[1].set_ylim([0, 200])
axes[1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('LenError_analysis.pdf')


# Plot 6: Deposition Speed vs Struts & Radius
dep_speeds = np.array([0.1, 0.4, 1])
p6_struts_created = np.array([480, 486, 493])
p6_radius = np.array([149-51, 152-51, 149-50])

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
#fig.suptitle('Deposition Speed vs Struts & Radius', fontsize=24)

axes[0].plot(dep_speeds, p6_struts_created, 's-')
axes[0].set_title('Struts vs Speed', fontsize=24)
axes[0].set_xlabel('Deposition Speed', fontsize=24)
axes[0].set_ylabel('Number of Struts', fontsize=24)
axes[0].tick_params(axis='both', labelsize=20)
axes[0].set_ylim([0, 1200])
axes[0].grid(True)

axes[1].plot(dep_speeds, p6_radius, 's-')
axes[1].set_title('Radius vs Speed', fontsize=24)
axes[1].set_xlabel('Deposition Speed', fontsize=24)
axes[1].set_ylabel('Radius', fontsize=24)
axes[1].tick_params(axis='both', labelsize=20)
axes[1].set_ylim([0, 200])
axes[1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('DepSpeed_analysis.pdf')
plt.show()
