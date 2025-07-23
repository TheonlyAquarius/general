import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict

class GRPOTrainer:
    """
    Manages the training loop for SynthNet using a GRPO-weighted 
    denoising diffusion objective.
    """

    def __init__(
        self,
        synthnet_model: nn.Module,
        optimizer: optim.Optimizer,
        num_denoising_steps: int,
        device: torch.device = torch.device("cpu")
    ):
        """
        Initializes the trainer.

        Args:
            synthnet_model (nn.Module): The model to be trained (your SynthNet architecture).
                                       It is assumed to have a forward pass signature of:
                                       forward(noisy_weights: Dict, timestep: int) -> Dict
            optimizer (optim.Optimizer): The optimizer for SynthNet's parameters.
            num_denoising_steps (int): The total number of steps in the denoising trajectory (T).
            device (torch.device): The device to run computations on.
        """
        self.synthnet_model = synthnet_model.to(device)
        self.optimizer = optimizer
        self.num_denoising_steps = num_denoising_steps
        self.device = device
        self.mse_loss = nn.MSELoss(reduction='sum') # Using sum to represent the L2 norm squared.

    def _evaluate_trajectory(self, trajectory: List[Dict[str, torch.Tensor]]) -> List[float]:
        """
        Placeholder for the user-defined evaluation function.

        This function must take the list of weight checkpoints (the trajectory)
        and return a list of scalar rewards (e.g., accuracy). The higher the
        reward, the better the performance of the weights at that step.

        Args:
            trajectory (List[Dict[str, torch.Tensor]]): A list of weight dictionaries,
                                                       where each dictionary represents
                                                       a state W_t.

        Returns:
            List[float]: A list of scalar rewards, one for each checkpoint.
        """
        # ======================================================================
        # USER IMPLEMENTATION REQUIRED HERE
        #
        # Example Logic:
        # rewards = []
        # for weights_checkpoint in trajectory:
        #     target_model.load_state_dict(weights_checkpoint)
        #     accuracy = evaluate_model_on_validation_set(target_model)
        #     rewards.append(accuracy)
        # return rewards
        # ======================================================================
        
        # For demonstration purposes, returning random rewards.
        # In a real scenario, this would be your actual evaluation logic.
        print("Warning: Using placeholder evaluation. Rewards are random.")
        return [torch.rand(1).item() for _ in trajectory]

    def train_step(self, w_0: Dict[str, torch.Tensor], w_t_initial: Dict[str, torch.Tensor]):
        """
        Performs a single, complete training step.

        Args:
            w_0 (Dict[str, torch.Tensor]): The target clean weights (ground truth).
            w_t_initial (Dict[str, torch.Tensor]): The initial noisy weights at step T.
        """
        self.synthnet_model.train()
        self.optimizer.zero_grad()

        # --- 1. Generate Trajectory ---
        trajectory = []
        w_current = {name: p.clone().to(self.device) for name, p in w_t_initial.items()}

        with torch.no_grad(): # Trajectory generation does not require gradients
            for t in reversed(range(self.num_denoising_steps)):
                timestep = torch.tensor([t], device=self.device)
                
                # The denoiser predicts the clean weights w_0 from the noisy w_t
                predicted_w_0 = self.synthnet_model(w_current, timestep)
                
                # This is a simplified DDPM-style update rule.
                # In a real implementation, you would use a proper scheduler
                # to derive w_{t-1} from the predicted w_0 and w_t.
                # For this example, we'll just use the prediction as the next state.
                w_current = predicted_w_0
                
                # Store a copy of the weights at this step
                trajectory.append({name: p.clone() for name, p in w_current.items()})

        # --- 2. Evaluate Checkpoints ---
        rewards = self._evaluate_trajectory(trajectory)
        rewards_tensor = torch.tensor(rewards, device=self.device)

        # --- 3. Calculate GRPO Advantage ---
        if len(rewards_tensor) > 1:
            group_mean = rewards_tensor.mean()
            group_std = rewards_tensor.std()
            # Add epsilon to prevent division by zero for groups with no variance
            advantages = (rewards_tensor - group_mean) / (group_std + 1e-8)
        else:
            # Handle the case of a single-element group
            advantages = torch.tensor([0.0], device=self.device)

        # --- 4. Compute Weighted Loss ---
        total_loss = 0.0
        # We iterate through the trajectory again, this time building the computation graph
        
        w_current_for_grad = {name: p.clone().to(self.device) for name, p in w_t_initial.items()}

        for t in reversed(range(self.num_denoising_steps)):
            # Detach to ensure the gradient is local to this specific step
            w_input_for_grad = {name: p.detach().requires_grad_(True) for name, p in w_current_for_grad.items()}
            
            timestep = torch.tensor([t], device=self.device)
            predicted_w_0 = self.synthnet_model(w_input_for_grad, timestep)

            # Calculate the local diffusion loss
            # This requires summing the MSE over all tensors in the weight dictionary
            local_diffusion_loss = 0
            for name in w_0.keys():
                local_diffusion_loss += self.mse_loss(predicted_w_0[name], w_0[name].to(self.device))
            
            # Retrieve the corresponding advantage
            # The trajectory index is (T-1) - t
            advantage = advantages[(self.num_denoising_steps - 1) - t]

            # Weight the local loss by the negative advantage
            step_loss = -advantage * local_diffusion_loss
            total_loss += step_loss
            
            # Update the state for the next iteration (without gradient tracking)
            w_current_for_grad = {name: p.data for name, p in predicted_w_0.items()}


        # --- 5. Perform Update ---
        # Normalize the loss by the number of steps to keep gradients stable
        if self.num_denoising_steps > 0:
            (total_loss / self.num_denoising_steps).backward()
        
        # Optional: Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.synthnet_model.parameters(), max_norm=1.0)
        
        self.optimizer.step()

        return total_loss.item()