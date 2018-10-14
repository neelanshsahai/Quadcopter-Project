import random
from collections import namedtuple, deque


class ReplayBuffer:
	"""A Model class to store some of the past experiences of the model in the form of [S, A, R, S']"""

	def __init__(self, buffer_size, batch_size):
		"""Initializes the Buffer Object and globaly declares the Deque, the NamedTuple and the BatchSize"""
		self.batch_size = batch_size
		self.memory = deque(maxlen=buffer_size)
		self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

	def add(self, state, action, reward, next_state, done):
		"""Add a new experience of the Agent to the MemoryBuffer"""
		e = self.experience(state, action, reward, next_state, done)
		self.memory.append(e)

	def sample(self, batch_size=64):
		"""Returns a random sample from the Memory of Experiences as a Batch of the specified size"""
		return random.sample(self.memory, k=self.batch_size)

	def __len__(self):
		"""Returns the Current size of the MemoryBuffer"""
		return len(self.memory)
