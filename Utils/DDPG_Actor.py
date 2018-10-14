from keras import layers, models, optimizers
from keras import backend as K


class Actor:
    """Actor Policy Model"""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize Parameters and Build Model"""
        self.state_size = state_size			# Dimension of Each State
        self.action_size = action_size			# Dimension of Each Action
        self.action_low = action_low			# Min Value of each Action Dimension
        self.action_high = action_high			# Max Value of each Action Dimension
        self.action_range = self.action_high - self.action_low

        self.build_model()

    def build_model(self):
        """An Actor (Policy) network that maps States to Actions"""

        # Define Input Layer
        states = layers.Input(shape=(self.state_size), name='states')

        # Hidden Layers
        net = layers.Dense(units=32, activation='relu')(states)
        net = layers.Dense(units=64, activation='relu')(net)
        net = layers.Dense(units=32, activation='relu')(net)

        # Final Output Layer with Sigmoid Activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)

        # Scaling the output for each action dimension
        actions = layers.Lambda(lambda x: (x*self.action_range)+self.action_low, name='actions')(raw_actions)

        # Create Keras Model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define Loss Function using Action Value (Q-Value) Gradient
        action_gradients = layers.Input(shape=self.action_size)
        loss = K.mean(x=(-action_gradients * actions))

        # Optimizer and Training Functions
        optimizer = optimizers.Adam(lr=0.0001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)

        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op
        )
