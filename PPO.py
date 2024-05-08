import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from core import S_INFO, S_LEN, A_DIM
################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
#         super(ActorCritic, self).__init__()

#         self.has_continuous_action_space = has_continuous_action_space

#         if has_continuous_action_space:
#             self.action_dim = action_dim
#             self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
#         # actor
#         if has_continuous_action_space:
#             self.actor = nn.Sequential(
#                 nn.Linear(state_dim, 128),
#                 nn.Tanh(),
#                 nn.Linear(128, 128),
#                 nn.Tanh(),
#                 nn.Linear(128, action_dim),
#                 nn.Tanh()
#             )
#         else:
#             self.actor = nn.Sequential(
#                 nn.Linear(state_dim, 64),
#                 nn.Tanh(),
#                 nn.Linear(64, 64),
#                 nn.Tanh(),
#                 nn.Linear(64, action_dim),
#                 nn.Softmax(dim=-1)
#             )
#         # critic
#         self.critic = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             nn.Tanh(),
#             nn.Linear(128, 128),
#             nn.Tanh(),
#             nn.Linear(128, 1)
#         )

#     def set_action_std(self, new_action_std):
#         if self.has_continuous_action_space:
#             self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
#         else:
#             print("--------------------------------------------------------------------------------------------")
#             print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
#             print("--------------------------------------------------------------------------------------------")

#     def forward(self):
#         raise NotImplementedError

#     def act(self, state):

#         if self.has_continuous_action_space:
#             action_mean = self.actor(state)
#             cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
#             dist = MultivariateNormal(action_mean, cov_mat)
#         else:
#             action_probs = self.actor(state)
#             dist = Categorical(action_probs)

#         action = dist.sample()
#         action_logprob = dist.log_prob(action)
#         state_val = self.critic(state)

#         return action.detach(), action_logprob.detach(), state_val.detach()

#     def evaluate(self, state, action):

#         if self.has_continuous_action_space:
#             action_mean = self.actor(state)

#             action_var = self.action_var.expand_as(action_mean)
#             cov_mat = torch.diag_embed(action_var).to(device)
#             dist = MultivariateNormal(action_mean, cov_mat)

#             # For Single Action Environments.
#             if self.action_dim == 1:
#                 action = action.reshape(-1, self.action_dim)
#         else:
#             action_probs = self.actor(state)
#             dist = Categorical(action_probs)
#         action_logprobs = dist.log_prob(action)
#         dist_entropy = dist.entropy()
#         state_values = self.critic(state)

#         return action_logprobs, state_values, dist_entropy
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # CNN layers for processing video features (SI and TI)
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        #self.conv_fc = nn.Linear(64 * S_LEN, 128)  # Adjust the input size of the fully connected layer

        # CNN layers for processing Trace feature (Value and edges) histrogram
        self.conv3 = nn.Conv1d(2, 32 ,kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        #self.conv_f2 = nn.Linear(64 * S_LEN, 128)  # Adjust the input size of the fully connected layer 

        # Fully connected layers for processing GLCM features
        self.fc_glcm = nn.Linear(6, 64)

        # LSTM for processing temporal features
        self.lstm = nn.LSTM(64+64+5, 128, batch_first=True)
        
        # Fully connected layers for lstm output
        self.fc_simulation = nn.Linear(128, 256)
        self.fc_simulation2 = nn.Linear(256, 128)




        # Actor and critic networks
        self.actor = nn.Sequential(
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self, state):
        # Reshape the state to match the expected dimensions
        state = state.view(-1, S_INFO, S_LEN)
        # Process streaming simulation features using LSTM
        simulation_ft = state[:, 0:5, :]
        glcm_features = state[:, 7:13 ,:]
        trace_ft = state[:, 13: ,:]
        # Process video features using CNN
        si_ti_features = state[:, 5:7, :] #[batch_size, 2, S_LEN]
        #print('Video feaures shape',video_features.shape)
        x_siti = torch.relu(self.conv1(si_ti_features))
        x_siti = torch.relu(self.conv2(x_siti))
        # x_siti = x_siti.view(x_siti.size(0), -1)
        # x_siti = torch.relu(self.conv_fc(x_siti))
        # Process GLCM features using fully connected layers
        x_glcm = torch.relu(self.fc_glcm(glcm_features))
        # Network condition feature
        #print('Trace feaures shape',trace_ft.shape)
        x_trace = self.conv3(trace_ft)
        x_trace = self.conv4(x_trace)

        x_cat = torch.cat([x_siti, x_trace,simulation_ft], dim=1)
        print('x_cat shape',x_cat.shape)
        x_cat = x_cat.view(-1, S_LEN, 133)
        x_lstm, _ = self.lstm(x_cat)
        print('x_lstm shape',x_lstm[0].shape)
        x_lstm_out= x_lstm[:,-1,:]
        x = torch.relu(self.fc_simulation(x_lstm_out))
        x = torch.relu(self.fc_simulation2(x))


        # Concatenate the processed features
        # print('x_video shape',x_video.shape)
        # print('x_temporal shape',x_temporal.shape)
        # print('x_glcm shape',x_glcm.shape)
        # print('x_network shape',x_network.shape)
        # print('x_edge shape',x_edge.shape)
        #x = torch.cat([x_video,x_trace, x_temporal, x_glcm],dim=1)
        #x = torch.cat([x_video,x_trace, x_temporal, x_glcm, x_network, x_edge], dim=1)

        return x
    def act(self, state):
        processed_state = self.forward(state)

        if self.has_continuous_action_space:
            action_mean = self.actor(processed_state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(processed_state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(processed_state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        processed_state = self.forward(state)

        if self.has_continuous_action_space:
            action_mean = self.actor(processed_state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(processed_state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(processed_state)

        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            action = action.clamp(-1, 1)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
