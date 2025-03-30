import random
import numpy as np
from collections import deque

class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self.beta_start = beta_start  # Start value of beta
        self.beta_frames = beta_frames  # Schedule length for beta
        self.frame = 1  # For beta calculation
        self.buffer = []
        self.priorities = []
        self.position = 0

    def beta_by_frame(self, frame_idx):
        """Calculates beta value for importance-sampling weights"""
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def add(self, experience, error=1.0):
        """Add new experience to memory with maximum priority"""
        max_priority = max(self.priorities) if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=32, beta=None):
        """Sample a batch of experiences with importance sampling weights"""
        if len(self.buffer) == 0:
            return [], [], []

        if beta is None:
            beta = self.beta_by_frame(self.frame)
            self.frame += 1
            
        # Convert priorities to sampling probabilities
        priorities = np.array(self.priorities, dtype=np.float32)
        scaled_p = priorities ** self.alpha
        sample_probs = scaled_p / scaled_p.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=sample_probs, replace=False)
        
        # Get selected experiences
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * sample_probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights
        
        return indices, samples, weights

    def update_priorities(self, batch_indices, errors, offset=1e-5):
        """Update priorities based on TD errors"""
        for i, idx in enumerate(batch_indices):
            self.priorities[idx] = abs(errors[i]) + offset  # Prevent zero priority

    def __len__(self):
        return len(self.buffer)
