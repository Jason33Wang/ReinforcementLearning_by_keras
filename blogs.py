import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
import keras
from keras.optimizers import RMSprop ,adam
from keras import backend as K

np.random.seed(1)


class DQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0

        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self._build_net()

    def target_replace_op(self):
        v1 = self.model2.get_weights()
        self.model1.set_weights(v1)
        print("params has changed")

    def _build_net(self):
        # 构建evaluation网络
        eval_inputs = Input(shape=(self.n_features,))
        x = Dense(64, activation='relu')(eval_inputs)
        x = Dense(64, activation='relu')(x)
        self.q_eval = Dense(self.n_actions)(x)
        # 构建target网络，注意这个target层输出是q_next而不是，算法中的q_target
        target_inputs = Input(shape=(self.n_features,))
        x = Dense(64, activation='relu')(target_inputs)
        x = Dense(64, activation='relu')(x)
        self.q_next = Dense(self.n_actions)(x)

        self.model1 = Model(target_inputs, self.q_next)
        self.model2 = Model(eval_inputs, self.q_eval)
        rmsprop = RMSprop(lr=self.lr)
        self.model1.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])
        self.model2.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition  # memory是一个二维列表
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = np.array(observation)
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            actions_value = self.model1.predict(observation)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_replace_op()
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        q_next, q_eval = self.model1.predict(batch_memory[:, -self.n_features:]), self.model2.predict(
            batch_memory[:, :self.n_features])
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        self.model2.fit(batch_memory[:, :self.n_features], q_target, epochs=10)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
