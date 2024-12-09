import matplotlib.pyplot as plt

# Log data
log_data = [
    # [1, 3.51, 122.64804469273743, 1553.3, 256.0, 4.9, 0.3, 179, 1.0000],
    [2, 4.51, 35.75957120980092, 311.3, 165.4, 3.0, 0.9, 653, 0.5835],
    [4, 6.12, 95.87665198237886, 1171.0, 318.8, 5.0, 0.5, 227, 0.9956],
    [8, 9.13, 116.88359788359789, 1462.8, 434.4, 4.9, 0.4, 189, 0.9947],
    [16, 12.60, 115.53157894736842, 1442.0, 413.6, 4.9, 0.4, 190, 0.9947],
    [32, 17.43, 118.02688172043011, 1477.7, 402.3, 4.9, 0.3, 186, 1.0000],
    [64, 27.92, 113.48484848484848, 1417.3, 341.3, 4.9, 0.3, 198, 1.0000],
    [128, 48.31, 112.6974358974359, 1404.0, 353.5, 4.9, 0.4, 195, 1.0000],
    [256, 88.08, 110.88265306122449, 1384.6, 326.8, 4.9, 0.4, 196, 1.0000],
    [512, 164.93, 112.16666666666667, 1384.6, 326.8, 4.9, 0.4, 196, 1.0000]
]


# Lists to store the results
num_denoising_steps_list = []
avg_traj_length_list = []
avg_episode_reward_list = []
avg_episode_reward_std_list = []
avg_best_reward_list = []
avg_best_reward_std_list = []
success_rate_list = []

# Extract data from log_data
for entry in log_data:
    num_denoising_steps_list.append(entry[0])
    avg_traj_length_list.append(entry[2])
    avg_episode_reward_list.append(entry[3])
    avg_episode_reward_std_list.append(entry[4])
    avg_best_reward_list.append(entry[5])
    avg_best_reward_std_list.append(entry[6])
    success_rate_list.append(entry[8])

# Plotting
plt.figure(figsize=(12, 8))

# Plot average episode reward with shading
plt.subplot(2, 2, 1)
plt.semilogx(num_denoising_steps_list, avg_episode_reward_list, marker='o', label='Avg Episode Reward', color='b')
plt.fill_between(num_denoising_steps_list,
                 [avg_episode - std for avg_episode, std in zip(avg_episode_reward_list, avg_episode_reward_std_list)],
                 [avg_episode + std for avg_episode, std in zip(avg_episode_reward_list, avg_episode_reward_std_list)],
                 color='b', alpha=0.2, label='Std Dev')
plt.title('Average Episode Reward vs. Number of Denoising Steps')
plt.xlabel('Number of Denoising Steps')
plt.ylabel('Average Episode Reward')
plt.grid(True)
plt.legend()

# Plot average best reward with shading
plt.subplot(2, 2, 2)
plt.semilogx(num_denoising_steps_list, avg_best_reward_list, marker='o', label='Avg Best Reward', color='g')
plt.fill_between(num_denoising_steps_list,
                 [avg_best - std for avg_best, std in zip(avg_best_reward_list, avg_best_reward_std_list)],
                 [avg_best + std for avg_best, std in zip(avg_best_reward_list, avg_best_reward_std_list)],
                 color='g', alpha=0.2, label='Std Dev')
plt.title('Average Best Reward vs. Number of Denoising Steps')
plt.xlabel('Number of Denoising Steps')
plt.ylabel('Average Best Reward')
plt.grid(True)
plt.legend()

# Plot average trajectory length
plt.subplot(2, 2, 3)
plt.semilogx(num_denoising_steps_list, avg_traj_length_list, marker='o', label='Avg Trajectory Length', color='r')
plt.title('Average Trajectory Length vs. Number of Denoising Steps')
plt.xlabel('Number of Denoising Steps')
plt.ylabel('Average Trajectory Length')
plt.grid(True)
plt.legend()

# Plot success rate
plt.subplot(2, 2, 4)
plt.semilogx(num_denoising_steps_list, success_rate_list, marker='o', label='Success Rate', color='m')
plt.title('Success Rate vs. Number of Denoising Steps')
plt.xlabel('Number of Denoising Steps')
plt.ylabel('Success Rate')
plt.grid(True)
plt.legend()

plt.suptitle('Flow-matching policy with various denoising steps')
# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig('./test_denoise.png')
plt.close()  # Close the figure to free up memory
