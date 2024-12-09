def plot_eval_statistics(eval_statistics, log_dir:str):
        num_denoising_steps_list, duration_list, avg_traj_length_list, avg_episode_reward_list, avg_best_reward_list, avg_episode_reward_std_list, avg_best_reward_std_list, success_rate_list, num_episodes_finished_list = eval_statistics
        import matplotlib.pyplot as plt
        import os
        # Plotting
        plt.figure(figsize=(12, 8))

        # Plot average episode reward with shading
        plt.subplot(2, 3, 1)
        plt.semilogx(num_denoising_steps_list, avg_episode_reward_list, marker='o', label='Avg Episode Reward', color='b')
        plt.fill_between(num_denoising_steps_list,
                    [avg_episode - std for avg_episode, std in zip(avg_episode_reward_list, avg_episode_reward_std_list)],
                    [avg_episode + std for avg_episode, std in zip(avg_episode_reward_list, avg_episode_reward_std_list)],
                    color='b', alpha=0.2, label='Std Dev')
        plt.axvline(x=20,  color='black', linestyle='--', label='Training Steps')
        plt.title('Average Episode Reward ')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Average Episode Reward')
        plt.grid(True)
        plt.legend()

        # Plot average trajectory length
        plt.subplot(2, 3, 2)
        plt.semilogx(num_denoising_steps_list, avg_traj_length_list, marker='o', label='Avg Trajectory Length', color='r')
        plt.axvline(x=20,  color='black', linestyle='--', label='Training Steps')
        plt.title('Average Trajectory Length ')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Average Trajectory Length')
        plt.grid(True)
        plt.legend()

        # Plot average best reward with shading
        plt.subplot(2, 3, 4)
        plt.semilogx(num_denoising_steps_list, avg_best_reward_list, marker='o', label='Avg Best Reward', color='g')
        plt.fill_between(num_denoising_steps_list,
                    [avg_best - std for avg_best, std in zip(avg_best_reward_list, avg_best_reward_std_list)],
                    [avg_best + std for avg_best, std in zip(avg_best_reward_list, avg_best_reward_std_list)],
                    color='g', alpha=0.2, label='Std Dev')
        plt.axvline(x=20,  color='black', linestyle='--', label='Training Steps')
        plt.title('Average Best Reward ')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Average Best Reward')
        plt.grid(True)
        plt.legend()

        # Plot success rate
        plt.subplot(2, 3, 5)
        plt.semilogx(num_denoising_steps_list, success_rate_list, marker='o', label='Success Rate', color='y')
        plt.axvline(x=20,  color='black', linestyle='--', label='Training Steps')
        plt.title('Success Rate ')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Success Rate')
        plt.grid(True)
        plt.legend()

        # Plot inference time
        plt.subplot(2, 3, 3)
        plt.semilogx(num_denoising_steps_list, duration_list, marker='o', label='Duration', color='brown')
        plt.axvline(x=20,  color='black', linestyle='--', label='Training Steps')
        plt.title('Inference Duration ')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Inference Duration')
        plt.grid(True)
        plt.legend()

        ''' # this is the number of episodes in the trial, which is inversly related to the trajectory length. kind of like redundant information. 
        plt.subplot(2, 3, 6)
        plt.semilogx(num_denoising_steps_list, num_episodes_finished_list, marker='o', label='Duration', color='skyblue')
        plt.axvline(x=20,  color='black', linestyle='--', label='Training Steps')
        plt.title('Finished Episodes ')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Finished Episodes')
        plt.grid(True)
        plt.legend()
        '''
        plt.suptitle('Diffusion Policy with Varying Denoising Steps', fontsize=25)
        plt.tight_layout()

        fig_path =os.path.join(log_dir, f'denoise_step.png')
        plt.savefig(fig_path)
        print(f"figure saved to {fig_path}")
        plt.close()  # Close the figure to free up memory


def read_eval_statistics(npz_file_path):
        import numpy as np
        import os

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
        
        # return all these list
        eval_statistics=(num_denoising_steps_list, duration_list, avg_traj_length_list, avg_episode_reward_list, avg_best_reward_list, avg_episode_reward_std_list, avg_best_reward_std_list, success_rate_list, num_episodes_finished_list)
        return eval_statistics
    
    
if __name__=="__main__":
    # Example usage 
    npz_file_path = 'eval_statistics_diffusion.npz'
    eval_statistics = read_eval_statistics(npz_file_path)
    plot_eval_statistics(eval_statistics, log_dir='./plot')