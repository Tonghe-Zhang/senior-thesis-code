import numpy as np
import matplotlib.pyplot as plt
import os

def read_eval_statistics(npz_file_path):
    # Load the .npz file
    loaded_data = np.load(npz_file_path)

    # Extract the structured array
    data = loaded_data['data']

    # Extract individual lists
    num_denoising_steps_list = data['num_denoising_steps']
    duration_list = data['duration']
    avg_traj_length_list = data['avg_traj_length']
    avg_episode_reward_list = data['avg_episode_reward']
    avg_best_reward_list = data['avg_best_reward']
    avg_episode_reward_std_list = data['avg_episode_reward_std']
    avg_best_reward_std_list = data['avg_best_reward_std']
    success_rate_list = data['success_rate']
    num_episodes_finished_list = data['num_episodes_finished']

    # Return all these lists
    eval_statistics = (num_denoising_steps_list, duration_list, avg_traj_length_list, avg_episode_reward_list, avg_best_reward_list, avg_episode_reward_std_list, avg_best_reward_std_list, success_rate_list, num_episodes_finished_list)
    return eval_statistics

def plot_eval_statistics(eval_statistics_1, eval_statistics_2, labels):
    num_denoising_steps_list_1, duration_list_1, avg_traj_length_list_1, avg_episode_reward_list_1, avg_best_reward_list_1, avg_episode_reward_std_list_1, avg_best_reward_std_list_1, success_rate_list_1, num_episodes_finished_list_1 = eval_statistics_1
    num_denoising_steps_list_2, duration_list_2, avg_traj_length_list_2, avg_episode_reward_list_2, avg_best_reward_list_2, avg_episode_reward_std_list_2, avg_best_reward_std_list_2, success_rate_list_2, num_episodes_finished_list_2 = eval_statistics_2

    # Plotting
    plt.figure(figsize=(18, 12))

    # Plot inference duration
    plt.subplot(2, 2, 1)
    plt.semilogx(num_denoising_steps_list_1, duration_list_1, marker='o', label=f'{labels[0]} Duration', color='b')
    plt.semilogx(num_denoising_steps_list_2, duration_list_2, marker='x', label=f'{labels[1]} Duration', color='r')
    plt.title('Inference Duration')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Inference Duration')
    plt.grid(True)
    plt.legend()

    # Plot average trajectory length
    plt.subplot(2, 2, 2)
    plt.semilogx(num_denoising_steps_list_1, avg_traj_length_list_1, marker='o', label=f'{labels[0]} Avg Trajectory Length', color='b')
    plt.semilogx(num_denoising_steps_list_2, avg_traj_length_list_2, marker='x', label=f'{labels[1]} Avg Trajectory Length', color='r')
    plt.title('Average Trajectory Length')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Average Trajectory Length')
    plt.grid(True)
    plt.legend()

    # Plot average episode reward with shading
    plt.subplot(2, 2, 3)
    plt.semilogx(num_denoising_steps_list_1, avg_episode_reward_list_1, marker='o', label=f'{labels[0]} Avg Episode Reward', color='b')
    plt.fill_between(num_denoising_steps_list_1,
                     [avg_episode - std for avg_episode, std in zip(avg_episode_reward_list_1, avg_episode_reward_std_list_1)],
                     [avg_episode + std for avg_episode, std in zip(avg_episode_reward_list_1, avg_episode_reward_std_list_1)],
                     color='b', alpha=0.2)
    plt.semilogx(num_denoising_steps_list_2, avg_episode_reward_list_2, marker='x', label=f'{labels[1]} Avg Episode Reward', color='r')
    plt.fill_between(num_denoising_steps_list_2,
                     [avg_episode - std for avg_episode, std in zip(avg_episode_reward_list_2, avg_episode_reward_std_list_2)],
                     [avg_episode + std for avg_episode, std in zip(avg_episode_reward_list_2, avg_episode_reward_std_list_2)],
                     color='r', alpha=0.2)
    plt.title('Average Episode Reward')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Average Episode Reward')
    plt.grid(True)
    plt.legend()

    plt.suptitle('Comparison of Models', fontsize=25)
    plt.tight_layout()
    
    fig_path = 'model_comparison.png'
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")
    plt.close()  # Close the figure to free up memory

if __name__ == "__main__":
    statistics_flow = read_eval_statistics('eval_statistics_reflow.npz')
    statistics_shortcut_flow = read_eval_statistics('eval_statistics_shortcut_flow.npz')
    labels = ['Flow Model', 'Shortcut Flow Model']
    plot_eval_statistics(statistics_flow, statistics_shortcut_flow, labels)
