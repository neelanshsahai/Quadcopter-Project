from keras import layers, models, optimizers, regularizers
from keras import backend as K


class Critic:
    """Critic (Value) Model Class"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    def build_model(self):
        """Build a critic network that maps (state, action) pairs to Q-Values"""
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Hidden Layers for state pathway
        net_states = layers.Dense(units=320, kernel_regularizer=regularizers.l2(0.01), activation='relu')(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.25)(net_states)
        net_states = layers.Dense(units=160, kernel_regularizer=regularizers.l2(0.01), activation='relu')(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.25)(net_states)
        net_states = layers.Dense(units=80, kernel_regularizer=regularizers.l2(0.01), activation='relu')(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.25)(net_states)
        net_states = layers.Dense(units=40, kernel_regularizer=regularizers.l2(0.01), activation='relu')(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.25)(net_states)

        # Hidden Layer for action pathway
        net_actions = layers.Dense(units=320, kernel_regularizer=regularizers.l2(0.01), activation='relu')(actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(0.25)(net_actions)
        net_actions = layers.Dense(units=160, kernel_regularizer=regularizers.l2(0.01), activation='relu')(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(0.25)(net_actions)
        net_actions = layers.Dense(units=80, kernel_regularizer=regularizers.l2(0.01), activation='relu')(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(0.25)(net_actions)
        net_actions = layers.Dense(units=40, kernel_regularizer=regularizers.l2(0.01), activation='relu')(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(0.25)(net_actions)

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Final Output layer
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create a Keras Model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define Optimizer and Compile the Model
        optimizer = optimizers.Adam(lr=0.0001)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Action Gradients (derivative of Q_Value
        action_gradient = K.gradients(Q_values, actions)

        # Function to fetch action gradients
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradient
        )
