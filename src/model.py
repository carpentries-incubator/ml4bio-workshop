import pandas as pd
import numpy as np
import sklearn

class Model:
	def __init__(self, model):
		self.model = model

	def setName(self, name):
		self.name = name

	def setComment(self, comment):
		self.comment = comment

	def getName(self):
		return self.name

	def getComment(self):
		return self.comment

