import numpy as np


class Binomial:

	def __init__(self, K, T, S, sigma, r, N, q=0, option_type='european', exercise_type='call', barrier=None):

		self.strike = K
		self.maturity = T
		self.spot = S
		self.vol = sigma
		self.ir = r
		self.n_steps = N
		self.option_type = option_type
		self.exercise_type = exercise_type
		self.dividend = q
		self.barrier = barrier

		
	def build_lattice(self):

		self.nu = self.ir - self.dividend - 0.5*self.vol**2
		self.delta_t = self.maturity / float(self.n_steps)
		self.disc = np.exp(-self.ir * self.delta_t)
		self.delta_up = np.sqrt((self.vol**2)*self.delta_t + (self.nu**2)*(self.delta_t**2))
		self.delta_down = -self.delta_up
		self.p_up = 0.5 + 0.5*(self.nu*self.delta_t/float(self.delta_up))
		self.p_down = 1 - self.p_up


	def print_attr(self):

		print('nu is ' + str(self.nu))
		print('disc is ' + str(self.disc))
		print('delta_up is ' + str(self.delta_up))
		print('delta_down is ' + str(self.delta_down))
		print('delta_t is ' + str(self.delta_t))
		print('p_up is ' + str(self.p_up))
		print('p_down is ' + str(self.p_down))


	def compute_spot(self, t):
		""" Return spot price at time t in a (t+1) dimensional array """

		assert (t >= 0 and t <= self.n_steps)
		spot_price = np.zeros(t + 1)
		for i in range(t + 1):
			spot_price[i] = i*self.delta_up + (self.n_steps - i)*self.delta_down
		spot_price = self.spot * np.exp(spot_price)

		return spot_price


	def compute_option(self, s):
		""" Return option value given spot price vector s """

		if self.exercise_type == 'call':
			value = np.maximum(0, s - self.strike)
		else:
			value = np.maximum(0, self.strike - s)

		return value


	def vanilla_european(self):
		""" Compute price of vanilla European option """

		spot_at_maturity = self.compute_spot(self.n_steps)
		option_value = self.compute_option(spot_at_maturity)

		for i in range(self.n_steps, -1, -1):
			for j in range(i):
				option_value[j] = self.disc * (self.p_down*option_value[j] + self.p_up*option_value[j+1])

		return option_value[0]


	def vanilla_american(self):
		""" Compute price of vanilla American option """

		spot_price = self.compute_spot(self.n_steps)
		option_value = self.compute_option(spot_price)
		for i in range(self.n_steps, -1, -1):
			for j in range(i):
				spot_price[j] = spot_price[j] * np.exp(-self.delta_down)
				option_value[j] = self.disc * (self.p_down*option_value[j] + self.p_up*option_value[j+1])
				early_exericise = self.compute_option(spot_price[j])
				option_value[j] = np.maximum(early_exericise, option_value[j])

		return option_value[0]


	def down_and_out_european(self):
		""" Compute price of down-and-out American option """

		spot_price = self.compute_spot(self.n_steps)
		option_value = self.compute_option(spot_price)
		option_value[spot_price <= self.barrier] = 0

		for i in range(self.n_steps, -1, -1):
			for j in range(i):
				spot_price[j] = spot_price[j] * np.exp(-self.delta_down)

				if spot_price[j] > self.barrier:
					option_value[j] = self.disc * (self.p_down*option_value[j] + self.p_up*option_value[j+1])
				else: 
					option_value[j] = 0

		return option_value[0]


	def down_and_out_american(self):
		""" Compute price of down-and-out American option """

		spot_price = self.compute_spot(self.n_steps)
		option_value = self.compute_option(spot_price)
		option_value[spot_price <= self.barrier] = 0

		for i in range(self.n_steps, -1, -1):
			for j in range(i):
				spot_price[j] = spot_price[j] * np.exp(-self.delta_down)

				if spot_price[j] > self.barrier:
					option_value[j] = self.disc * (self.p_down*option_value[j] + self.p_up*option_value[j+1])
					early_exericise = self.compute_option(spot_price[j])
					option_value[j] = np.maximum(early_exericise, option_value[j])
				else: 
					option_value[j] = 0

		return option_value[0]



	def price(self):

		self.build_lattice()
		if self.option_type == 'vanilla european':
			return self.vanilla_european()

		elif self.option_type == 'vanilla american':
			return self.vanilla_american()

		elif self.option_type == 'down-and-out european':
			return self.down_and_out_european()

		elif self.option_type == 'down-and-out american':
			return self.down_and_out_american()




vanilla_european_call = Binomial(100,1,100,0.2,0.06,3,0,'vanilla european','call')
print(vanilla_european_call.price())

vanilla_american_put = Binomial(100,1,100,0.2,0.06,3,0,'vanilla american','put')
print(vanilla_american_put.price())

down_and_out_american_call = Binomial(100,1,100,0.2,0.06,3,0.01,'down-and-out european','call',95)
print(down_and_out_american_call.price())

down_and_out_american_call = Binomial(100,1,100,0.2,0.06,3,0,'down-and-out american','call',95)
print(down_and_out_american_call.price())



