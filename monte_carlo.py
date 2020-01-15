import numpy as np
import ghalton
from scipy.special import ndtri


class MonteCarlo:


	def __init__(self, K, T, S, sigma, r, N, M, q, option_type, exercise_type, 
		barrier=None, fixing=None, S2=None, sigma2=None, q2=None, 
		var_reduction=None,  quasi_monte_carlo=None,
		stochastic_vol=None, var_bar=None, alpha_var=None, sigma_var=None, rho_S_var=None, rho_S_S2=None, rho_S2_var=None, rho_S2_var2=None, rho_S_var2=None, rho_var1_var2=None, var2_bar=None, alpha2_var=None, sigma_var2=None):

		self.strike = K
		self.maturity = T
		self.spot = S
		self.vol = sigma
		self.ir = r
		self.n_steps = N
		self.n_paths = M
		self.option_type = option_type
		self.exercise_type = exercise_type
		self.dividend = q
		self.barrier = barrier
		self.fixing = fixing
		self.spot2 = S2
		self.vol2 = sigma2
		self.dividend2 = q2
		self.var_reduction = var_reduction
		self.quasi = quasi_monte_carlo
		self.stochastic_vol = stochastic_vol
		self.var_bar = var_bar
		self.alpha_var = alpha_var
		self.sigma_var = sigma_var
		self.rho_S_var = rho_S_var
		self.rho_S_S2 = rho_S_S2
		self.rho_S2_var = rho_S2_var
		self.rho_S2_var2 = rho_S2_var2
		self.rho_S_var2 = rho_S_var2
		self.rho_var1_var2 = rho_var1_var2
		self.var2_bar = var2_bar
		self.alpha2_var = alpha2_var 
		self.sigma_var2 = sigma_var2



	def sanity_check(self):

		assert (self.n_steps >= 1)
		assert (self.n_paths >= 100)


		if self.stochastic_vol == 'mrsr':
			assert(self.var_bar is not None and self.alpha_var is not None and 
				self.alpha_var is not None and rho_S_var is not None)

		if self.option_type == 'spread european':
			assert(self.S2 is not None and self.vol2 is not None and 
				self.q2 is not None and self.rho_S_S2 is not None)

			if self.stochastic_vol == 'mrsr':
				assert(rho_S2_var is not None and rho_S2_var2 is not None and 
					rho_S_var2 is not None and rho_var1_var2 is not None and 
					var2_bar is not None and alpha2_var is not None and sigma_var2 is not None)

		if self.option_type == 'down and out european':
			assert (self.barrier is not None)

		if self.option_type == 'arithmetic asian european' or self.option_type == 'lookback european':
			assert(self.fixing is not None and type(self.fixing) == list)
			for item in self.fixing:
				assert(type(item) == int and item >= 0 and item <= self.n_steps)


	def gaussian_samples(self):
		""" Generate standard Gaussian samples """

		z_mat = np.random.normal(0, 1, (self.n_steps, self.n_paths)) 
		return z_mat


	def quasi_gaussian_samples(self):
		""" Generate standard Gaussian samples using low-discrepancy Halton seq
			with inverse transform """

		g = ghalton.GeneralizedHalton(100)
		unif = np.array(g.get(int(self.n_steps * self.n_paths / 100) + 1)).flatten()
		unif = unif[:self.n_steps*self.n_paths]
		z_mat = ndtri(unif).reshape((self.n_steps, self.n_paths))

		return z_mat


	def multi_gaussian_samples(self):
		""" Generate multivariate Gaussian samples """

		mu = np.array([0,0,0,0])
		cov = np.array([[1, self.rho_S_S2, self.rho_S_var, self.rho_S_var2],
						[self.rho_S_S2, 1, self.rho_S2_var, self.rho_S2_var2],
						[self.rho_S_var, self.rho_S2_var, 1, self.rho_var1_var2],
						[self.rho_S_var2, self.rho_S2_var2, self.rho_var1_var2, 1]])

		multi_gaussian_mat = np.random.multivariate_normal(mu, cov, self.n_steps*self.n_paths)

		z_mat = multi_gaussian_mat[:,0].reshape((self.n_steps, self.n_paths))
		z2_mat = multi_gaussian_mat[:,1].reshape((self.n_steps, self.n_paths))
		zvar_mat = multi_gaussian_mat[:,2].reshape((self.n_steps, self.n_paths))
		zvar2_mat = multi_gaussian_mat[:,3].reshape((self.n_steps, self.n_paths))

		return (z_mat, z2_mat, zvar_mat, zvar2_mat)



	def initialize_params(self):
		""" Precompute key params """

		self.nu = self.ir - self.dividend - 0.5*self.vol**2
		self.delta_t = self.maturity / float(self.n_steps)

		if self.quasi == 'halton':
			self.z_mat = self.quasi_gaussian_samples()
		else:
			self.z_mat = self.gaussian_samples()

		if self.stochastic_vol == 'mrsr':
			self.zvar_mat = self.rho_S_var * self.z_mat + np.sqrt(1 - self.rho_S_var**2) * self.gaussian_samples()
		else:
			self.zvar_mat = None

		if self.option_type == 'spread european':
			self.nu2 = self.ir - self.dividend2 - 0.5*self.vol2**2

			if self.stochastic_vol == 'mrsr':
				self.z_mat, self.z2_mat, self.zvar_mat, self.zvar2_mat = self.multi_gaussian_samples()
			else:

				self.z2_mat = self.rho_S_S2 * self.z_mat + np.sqrt(1 - self.rho_S_S2**2) * self.gaussian_samples()
				self.zvar_mat, self.zvar2_mat = (None,None)	
		else:
			self.nu2 = None



	def gbm(self, z_mat, zvar_mat, S_0, nu, var_bar, alpha_var, rho_S_var):
		""" Similate geometric Brownian motion paths, optional stochastic vol using
			mean-reverting square root process """

		path_mat = np.zeros((self.n_paths, self.n_steps + 1))
		path_mat[:,0] = np.log(S_0)

		if self.stochastic_vol == 'mrsr':

			var_mat = np.zeros((self.n_paths, self.n_steps + 1))
			var_mat[:,0] = self.vol**2

			for i in range(self.n_steps):
				z = z_mat[i,:]
				zvar = zvar_mat[i,:]
				var_mat[:,i+1] = var_mat[:,i] + alpha_var * (var_bar - var_mat[:,i]) * self.delta_t + self.sigma_var * np.sqrt(var_mat[:,i]) * np.sqrt(self.delta_t) * zvar
				path_mat[:,i+1] = path_mat[:,i] + (self.ir - self.dividend - 0.5*var_mat[:,i+1]) * self.delta_t + np.sqrt(var_mat[:,i+1]) * np.sqrt(self.delta_t) * z

		else:

			for i in range(self.n_steps):
				z = z_mat[i,:]
				path_mat[:,i+1] = path_mat[:,i] + nu * self.delta_t + self.vol * np.sqrt(self.delta_t) * z

		path_mat = np.exp(path_mat)
		return path_mat



	def compute_option(self, s, K):
		""" Return option value given spot price vector s and strike K """

		if self.exercise_type == 'call':
			value = np.maximum(0, s - K)
		else:
			value = np.maximum(0, K - s)

		return value


	def vanilla_european(self):
		""" Compute price of vanilla European option, optional variance reduction
			using antithetic variables and control variates """

		path_mat = self.gbm(self.z_mat, self.zvar_mat, self.spot, self.nu, self.var_bar, self.alpha_var, self.rho_S_var)
		S_T = path_mat[:, self.n_steps]
		
		if self.var_reduction == 'antithetic':

			path_mat_prime = self.gbm(-self.z_mat, self.zvar_mat, self.spot, self.nu, self.var_bar, self.alpha_var, self.rho_S_var)
			S_T_prime = path_mat_prime[:, self.n_steps]
			C_T = np.mean(0.5*(self.compute_option(S_T, self.strike) + self.compute_option(S_T_prime, self.strike)))

		elif self.var_reduction == 'control':

			Y_T = self.compute_option(S_T, self.strike)
			X_T = S_T

			E_XT = np.exp(self.ir * self.maturity) * self.spot
			XT_bar = np.mean(X_T)
			YT_bar = np.mean(Y_T)
			lambda_star = np.mean((X_T - XT_bar)*(Y_T - YT_bar)) / np.mean((X_T - XT_bar)**2)

			C_T = np.mean(Y_T - lambda_star * (X_T - E_XT))

		else:
			C_T = np.mean(self.compute_option(S_T, self.strike))

		C_0 = C_T * np.exp(-self.ir * self.maturity)

		return C_0



	def spread_european(self):
		""" Compute price of vanilla European spread option on the difference between
			two stocks """

		path1_mat = self.gbm(self.z_mat, self.zvar_mat, self.spot, self.nu, self.var_bar, self.alpha_var, self.rho_S_var)
		path2_mat = self.gbm(self.z2_mat, self.zvar2_mat, self.spot2, self.nu2, self.var2_bar, self.alpha2_var, self.rho_S2_var2)

		S1_T = path1_mat[:, self.n_steps]
		S2_T = path2_mat[:, self.n_steps]

		C_T = np.mean(self.compute_option(S1_T - S2_T, self.strike))
		C_0 = C_T * np.exp(-self.ir * self.maturity)

		return C_0



	def down_and_out_european(self):
		""" Compute price of down-and-out European option """

		path_mat = self.gbm(self.z_mat, self.zvar_mat, self.spot, self.nu, self.var_bar, self.alpha_var, self.rho_S_var)
		C_T = np.zeros(self.n_paths)

		for i in range(self.n_paths):
			if np.min(path_mat[i,:]) > self.barrier:
				C_T[i] = self.compute_option(path_mat[i, self.n_steps], self.strike)

		C_T = np.mean(C_T)
		C_0 = C_T * np.exp(-self.ir * self.maturity)

		return C_0



	def arithmetic_asian_european(self):
		""" Compute price of fixed-strike arithmetic Asian European option """

		path_mat = self.gbm(self.z_mat, self.zvar_mat, self.spot, self.nu, self.var_bar, self.alpha_var, self.rho_S_var)
		arithmetic_avg = np.mean(path_mat[:, self.fixing], axis=1)
		C_T = np.mean(self.compute_option(arithmetic_avg, self.strike))
		C_0 = C_T * np.exp(-self.ir * self.maturity)

		return C_0



	def lookback_european(self):
		""" Compute price of floating-strike lookback European option """

		path_mat = self.gbm(self.z_mat, self.zvar_mat, self.spot, self.nu, self.var_bar, self.alpha_var, self.rho_S_var)
		S_T = path_mat[:, self.n_steps]
		peak = np.amax(path_mat[:, self.fixing], axis=1)
		C_T = np.mean(self.compute_option(peak, path_mat[:, self.n_steps]))
		C_0 = C_T * np.exp(-self.ir * self.maturity)

		return C_0



	def price(self):

		self.initialize_params()

		if self.option_type == 'vanilla european':
			return self.vanilla_european()

		elif self.option_type == 'spread european':
			return self.spread_european()

		elif self.option_type == 'down and out european':
			return self.down_and_out_european()

		elif self.option_type == 'arithmetic asian european':
			return self.arithmetic_asian_european()

		elif self.option_type == 'lookback european':
			return self.lookback_european()




vanilla_european_call = MonteCarlo(K=100,T=1,S=100,sigma=0.2,r=0.06,N=10,M=1000,q=0.03,option_type='vanilla european',exercise_type='call')
print(vanilla_european_call.price())

vanilla_european_call_with_antithetic = MonteCarlo(K=100,T=1,S=100,sigma=0.2,r=0.06,N=10,M=1000,q=0.03,option_type='vanilla european',exercise_type='call',var_reduction='antithetic')
print(vanilla_european_call_with_antithetic.price())

vanilla_european_call_stochastic_vol = MonteCarlo(K=100,T=1,S=100,sigma=0.2,r=0.06,N=10,M=1000,q=0.03,option_type='vanilla european',exercise_type='call',stochastic_vol='mrsr',var_bar=0.04,alpha_var=1,sigma_var=0.01,rho_S_var=0.5)
print(vanilla_european_call_stochastic_vol.price())

vanilla_european_call_with_antithetic_stochastic_vol = MonteCarlo(K=100,T=1,S=100,sigma=0.2,r=0.06,N=10,M=1000,q=0.03,option_type='vanilla european',exercise_type='call',var_reduction='antithetic',stochastic_vol='mrsr',var_bar=0.04,alpha_var=1,sigma_var=0.01,rho_S_var=0.5)
print(vanilla_european_call_with_antithetic_stochastic_vol.price())

vanilla_european_call_with_control = MonteCarlo(K=100,T=1,S=100,sigma=0.2,r=0.06,N=10,M=1000,q=0.03,option_type='vanilla european',exercise_type='call',var_reduction='control')
print(vanilla_european_call_with_control.price())

vanilla_european_european_call_with_control_stochastic_vol = MonteCarlo(K=100,T=1,S=100,sigma=0.2,r=0.06,N=10,M=1000,q=0.03,option_type='vanilla european',exercise_type='call',var_reduction='control',stochastic_vol='mrsr',var_bar=0.04,alpha_var=1,sigma_var=0.01,rho_S_var=0.5)
print(vanilla_european_european_call_with_control_stochastic_vol.price())

vanilla_spread_european_call = MonteCarlo(K=1,T=1,S=100,sigma=0.2,r=0.06,N=1,M=1000,q=0.03,option_type='spread european',exercise_type='call',S2=110,sigma2=0.2,q2=0.04,rho_S_S2=0.5)
print(vanilla_spread_european_call.price())

vanilla_spread_european_call_stochastic_vol = MonteCarlo(K=1,T=1,S=100,sigma=0.2,r=0.06,N=10,M=1000,q=0.03,option_type='spread european',exercise_type='call',S2=110,sigma2=0.2,q2=0.04,stochastic_vol='mrsr',var_bar=0.04,var2_bar=0.09,alpha_var=1,alpha2_var=2.0,sigma_var=0.05,sigma_var2=0.06,rho_S_var=0.2,rho_S_S2=0.5,rho_S_var2=0.01,rho_S2_var=0.01,rho_S2_var2=0.3,rho_var1_var2=0.3)
print(vanilla_spread_european_call_stochastic_vol.price())

vanilla_european_call_quasi = MonteCarlo(K=100,T=1,S=100,sigma=0.2,r=0.06,N=10,M=1000,q=0.03,option_type='vanilla european',exercise_type='call',quasi_monte_carlo='halton')
print(vanilla_european_call_quasi.price())

down_and_out_european_call = MonteCarlo(K=100,T=1,S=100,sigma=0.2,r=0.06,N=10,M=1000,q=0.03,option_type='down and out european',exercise_type='call',barrier=99)
print(down_and_out_european_call.price())

arithmetic_asian_european_call = MonteCarlo(K=100,T=1,S=100,sigma=0.2,r=0.06,N=10,M=1000,q=0.03,option_type='arithmetic asian european',exercise_type='call',fixing=[i for i in range(11)])
print(arithmetic_asian_european_call.price())

lookback_european_call = MonteCarlo(K=None,T=1,S=100,sigma=0.2,r=0.06,N=10,M=1000,q=0.03,option_type='lookback european',exercise_type='call',fixing=[i for i in range(11)])
print(lookback_european_call.price())

lookback_european_call_stochastic_vol = MonteCarlo(K=None,T=1,S=100,sigma=0.2,r=0.06,N=10,M=1000,q=0.03,option_type='lookback european',exercise_type='call',fixing=[i for i in range(11)],stochastic_vol='mrsr',var_bar=0.04,alpha_var=1,sigma_var=0.01,rho_S_var=0.5)
print(lookback_european_call_stochastic_vol.price())


