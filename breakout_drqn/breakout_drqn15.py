#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 2018

@author: Sooyoung Moon

atari breakout drqn
"""

from keras.layers.convolutional import Conv2D

from keras.layers import Dense, Flatten, LSTM, TimeDistributed

from keras.optimizers import RMSprop

from keras.models import Sequential

from skimage.transform import resize

from skimage.color import rgb2gray

from collections import deque

from keras import backend as K

import tensorflow as tf

import numpy as np

import random

import gym

#from keras.utils.training_utils import multi_gpu_model



EPISODES = 50000


# 브레이크아웃에서의 DRQN 에이전트

class DRQNAgent:

    def __init__(self, action_size):

        self.render = False

        self.load_model = False

        # 행동의 크기 정의

        self.action_size = action_size

        # DRQN 하이퍼파라미터

        self.epsilon = 1.

        self.epsilon_start, self.epsilon_end = 1.0, 0.1

        self.exploration_steps = 1000000.

        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                            / self.exploration_steps
        
        self.batch_size = 32

        self.train_start = 50000

        self.update_target_rate = 10000

        self.discount_factor = 0.99

        # 리플레이 메모리, 최대 크기 400000

        self.memory = deque(maxlen=400000)

        self.no_op_steps = 30

        # 모델과 타겟모델을 생성하고 타겟모델 초기화 및 gpu에 모델 할당
        
        self.model = self.build_model()
        #self.model = multi_gpu_model(self.model, gpus = 4)
        self.target_model = self.build_model()
        #self.target_model = multi_gpu_model(self.target_model, gpus = 4)

        self.update_target_model()
        
        self.optimizer = self.optimizer()

       

        # 텐서보드 설정

        self.sess = tf.InteractiveSession()

        K.set_session(self.sess)


        self.avg_q_max, self.avg_loss = 0, 0

        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()

        self.summary_writer = tf.summary.FileWriter(

            'summary/breakout_drqn15', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

 

        if self.load_model:

            self.model.load_weights("./save_model/breakout_drqn15.h5")

 

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장

    def append_sample(self, history, action, reward, next_history, dead):

        self.memory.append((history, action, reward, next_history, dead))

          

    # Huber Loss를 이용하기 위해 최적화 함수를 직접 정의

    def optimizer(self):

        a = K.placeholder(shape=(None,), dtype='int32')

        y = K.placeholder(shape=(None,), dtype='float32')

 

        prediction = self.model.output

 

        a_one_hot = K.one_hot(a, self.action_size)

        q_value = K.sum(prediction * a_one_hot, axis=1)

        error = K.abs(y - q_value)

 

        quadratic_part = K.clip(error, 0.0, 1.0)

        linear_part = error - quadratic_part

        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

 

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)

        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)

        train = K.function([self.model.input, a, y], [loss], updates=updates)

 

        return train

 

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성

    def build_model(self):

        model = Sequential()

        model.add(TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),

                                  input_shape=(10, 84, 84, 1)))

                                  #input_shape=(time_step, row, col, channels)
                                  
        model.add(TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), activation='relu')))
       
        model.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu')))
       
        model.add(TimeDistributed(Flatten()))

        model.add(LSTM(512))

        model.add(Dense(128, activation='relu'))

        model.add(Dense(self.action_size))

        model.summary()

        return model

 

 

    # 타겟 모델을 모델의 가중치로 업데이트

    def update_target_model(self):

        self.target_model.set_weights(self.model.get_weights())

 

    # 입실론 탐욕 정책으로 행동 선택

    def get_action(self, history):

        history = np.float32(history / 255.0)

        if np.random.rand() <= self.epsilon:

            return random.randrange(self.action_size)

        else:

            q_value = self.model.predict(history)

            return np.argmax(q_value[0])

 

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습

    def train_model(self):
        if self.epsilon > self.epsilon_end:

            self.epsilon -= self.epsilon_decay_step

 

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, 10, 84, 84, 1))

        next_history = np.zeros((self.batch_size, 10, 84, 84, 1))

        target = np.zeros((self.batch_size,))

        action, reward, dead = [], [], []

 

        for i in range(self.batch_size):

            history[i] = np.float32(mini_batch[i][0] / 255.)

            next_history[i] = np.float32(mini_batch[i][3] / 255.)

            action.append(mini_batch[i][1])

            reward.append(mini_batch[i][2])

            dead.append(mini_batch[i][4])

 

        target_value = self.target_model.predict(next_history)

 

 

        for i in range(self.batch_size):

            if dead[i]:

                target[i] = reward[i]

            else:

                target[i] = reward[i] + self.discount_factor * np.amax(target_value[i])

        loss = self.optimizer([history, action, target])

        self.avg_loss += loss[0]

 

    # 각 에피소드 당 학습 정보를 기록

    def setup_summary(self):

        episode_total_reward = tf.Variable(0.)

        episode_avg_max_q = tf.Variable(0.)

        episode_duration = tf.Variable(0.)

        episode_avg_loss = tf.Variable(0.)

 

        tf.summary.scalar('Total_Reward/Episode', episode_total_reward)

        tf.summary.scalar('Average_Max_Q/Episode', episode_avg_max_q)

        tf.summary.scalar('Duration/Episode', episode_duration)

        tf.summary.scalar('Average_Loss/Episode', episode_avg_loss)

 

        summary_vars = [episode_total_reward, episode_avg_max_q,

                        episode_duration, episode_avg_loss]

        summary_placeholders = [tf.placeholder(tf.float32) for _ in

                                range(len(summary_vars))]

        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in

                      range(len(summary_vars))]


        summary_op = tf.summary.merge_all()

        return summary_placeholders, update_ops, summary_op

 

 

# 학습속도를 높이기 위해 흑백화면으로 전처리

def pre_processing(observe):

    processed_observe = np.uint8(

        resize(rgb2gray(observe), ( 84, 84 ), mode='constant') * 255)

    return processed_observe

 

 

if __name__ == "__main__":

    # 환경과 DRQN 에이전트 생성

    env = gym.make('BreakoutDeterministic-v4')

    agent = DRQNAgent(action_size=3)

 

    scores, episodes, global_step = [], [], 0

 
   
    for e in range(EPISODES):

        done = False

        dead = False

 

        step, score, start_life = 0, 0, 5

        observe = env.reset()

 

        for _ in range(random.randint( 1, agent.no_op_steps )):

            observe, _, _, _ = env.step( 1 )

 

        state = pre_processing(observe)

        state = state.reshape( 84, 84, 1 )

        history = np.stack(( state, state, state, state, state, 
                            state, state, state, state, state ), axis=0) 
                    #( 10, 84, 84, 1 )

        history = np.reshape([history], ( 1, 10, 84, 84, 1 ))

 

        while not done:

            if agent.render:

                env.render()

            global_step += 1

            step += 1
 

            # 바로 전 4개의 상태로 행동을 선택

            action = agent.get_action(history)

            # 1: 정지, 2: 왼쪽, 3: 오른쪽

            if action == 0:

                real_action = 1

            elif action == 1:

                real_action = 2
            else:

                real_action = 3

 

            # 선택한 행동으로 환경에서 한 타임스텝 진행

            observe, reward, done, info = env.step(real_action)

            #reward = reward * 10

            # 각 타임스텝마다 상태 전처리

            next_state = pre_processing(observe)

            next_state = next_state.reshape( 1, 84, 84, 1 )

            next_history = next_state.reshape( 1, 1, 84, 84, 1 )

            next_history = np.append(next_history, history[:, :9 , :, :, :], axis=1)

            next_history = np.reshape([next_history],( 1, 10, 84, 84, 1 ))

           

            if start_life > info['ale.lives']:

                dead = True

                start_life = info['ale.lives']

 

            reward = np.clip( reward, -1., 1. )


            # 샘플 <s, a, r, s'>을 저장

            agent.append_sample( history, action, reward, next_history, dead )

 

            if len(agent.memory) >= agent.train_start:

                agent.train_model()

 

            # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트

            if global_step % agent.update_target_rate == 0:

                agent.update_target_model()

 

            score += reward

 

            if dead:

                dead = False

            else:

                history = next_history

 

            if done:

                # 각 에피소드 당 학습 정보를 기록

                if global_step > agent.train_start:

                    stats = [score, agent.avg_q_max / float(step), step,

                             agent.avg_loss / float(step)]

                    for i in range(len(stats)):

                        agent.sess.run(agent.update_ops[i], feed_dict={

                            agent.summary_placeholders[i]: float(stats[i])

                        })

                    summary_str = agent.sess.run(agent.summary_op)

                    agent.summary_writer.add_summary(summary_str, e + 1)

 

                print("episode:", e, "  score:", score, "  memory length:",

                      len(agent.memory), "  epsilon:", agent.epsilon,

                      "  global_step:", global_step, "  average_q:",

                      agent.avg_q_max / float(step), "  average loss:",

                      agent.avg_loss / float(step))

 

                agent.avg_q_max, agent.avg_loss = 0, 0

        # 1000 에피소드마다 모델 저장

        if e % 1000 == 0:

            agent.model.save_weights("./save_model/breakout_drqn15.h5")
