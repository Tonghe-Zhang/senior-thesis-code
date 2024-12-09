def make_indices(traj_lengths, horizon_steps):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint, also save the number of steps before it within the same trajectory
        """
        
        
        """ 
            start, num_before_start = self.indices[idx]
            end = start + self.horizon_steps
            states = self.states[(start - num_before_start) : (start + 1)]
            actions = self.actions[start:end]
            states = torch.stack(
                [
                    states[max(num_before_start - t, 0)]
                    for t in reversed(range(self.cond_steps))
                ]
            )  # more recent is at the end
            conditions = {"state": states}
        """
        
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            indices += [
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)
            ]
            cur_traj_index += traj_length
        return indices
        


if __name__=="__main__":
    traj_lengths=[10,10,10,10]
    
    horizon_steps=6

    indices = make_indices(traj_lengths, horizon_steps)

    cond_steps=4
    
    import torch   
    import numpy as np
    
    states=torch.tensor(np.arange(100))
    actions=torch.tensor(np.arange(100))

    start, num_before_start = (14,4)
    end = start + horizon_steps   #20
    states = states[(start - num_before_start) : (start + 1)]    #[10, 11, 12, 13, 14]
    
    actions = actions[start:end]   #[14, 15, 16, 17, 18,19]
    states = torch.stack(
        [
            states[max(num_before_start - t, 0)]
            for t in reversed(range(cond_steps))    #[cond_steps-1, cond_steps-2, ... 3,2,1,0]
        ]
    )  # more recent is at the end
    conditions = {"state": states}
    
    
    
    print(actions)
    
            
    # print(indices)