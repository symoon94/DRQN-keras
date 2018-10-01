# Deep Recurrent Q-Network (DRQN)
** Keras implementation of DRQN. This code needs to be fixed though it can be trained at the end. Please do not see this repository seriously.


## Atari-Breakout

### Main Idea?
#### CRNN model using 'TimeDistributed' wrapper for Conv2D and LSTM

    
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
   

![브레이크아웃](https://github.com/symoon94/DRQN/blob/master/breakout_drqn/image/544604897.58.png)

[Breakout DRQN Source Code](https://github.com/symoon94/DRQN/blob/master/breakout_drqn/breakout_drqn15.py)


#

##### Code Reference: 'breakout_dqn.py' at https://github.com/rlcode/reinforcement-learning-kr


