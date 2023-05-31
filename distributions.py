import numpy as np
import torch


class Normal:
    def __init__(self, mu, sigma2):
        self.mu = mu
        self.sigma2 = sigma2

    # log density
    def ln_prob(self, theta):
        return - 0.5 * (theta-self.mu) ** 2 / self.sigma2

    # grad log density
    def ln_prob_grad(self, theta):
        return - (theta - self.mu) / self.sigma2

    def ln_prob_ln_prob_grad(self, theta):
        return self.ln_prob(theta), self.ln_prob_grad(theta)

    def sample(self, n_samples):
        return np.random.normal(loc=self.mu, scale=self.sigma2, size=(n_samples))


class MixtureNormal:
    def __init__(self, means, variances, weights):
        self.means = means
        self.variances = variances
        self.weights = weights

    # log density
    def ln_prob(self, theta):
        component_exponents = list()
        for mean, sigma2 in zip(self.means, self.variances):
            component_exponents.append(-0.5 * (theta - mean) ** 2 / sigma2)
        component_probs = list()
        for weight, exponent in zip(self.weights, component_exponents):
            component_probs.append(weight * np.exp(exponent))
        return np.log(np.sum(component_probs, axis=0))

    # grad log density
    def ln_prob_grad(self, theta):
        component_exponents = list()
        for mean, sigma2 in zip(self.means, self.variances):
            component_exponents.append(-0.5 * (theta - mean) ** 2 / sigma2)
        component_probs = list()
        for weight, exponent in zip(self.weights, component_exponents):
            component_probs.append(weight * np.exp(exponent))
        component_prob_grads = list()
        for weight, exponent, mean, sigma2 in zip(self.weights, component_exponents, self.means, self.variances):
            component_prob_grads.append(-1 * weight * 1/sigma2 * (theta - mean) * np.exp(exponent))
        return np.sum(component_prob_grads, axis=0) / np.sum(component_probs, axis=0)

    # grad2 log density
    def ln_prob_grad_two(self, theta):
        component_exponents = list()
        for mean, sigma2 in zip(self.means, self.variances):
            component_exponents.append(-0.5 * (theta - mean) ** 2 / sigma2)
        component_probs = list()
        for weight, exponent in zip(self.weights, component_exponents):
            component_probs.append(weight * np.exp(exponent))
        component_prob_grads = list()
        for weight, exponent, mean, sigma2 in zip(self.weights, component_exponents, self.means, self.variances):
            component_prob_grads.append(-1 * weight * 1/sigma2 * (theta - mean) * np.exp(exponent))
        component_prob_grads2 = list()
        for weight, exponent, mean, sigma2 in zip(self.weights, component_exponents, self.means, self.variances):
            component_prob_grads2.append(weight * (1/sigma2)**2 * (theta - mean)**2 * np.exp(exponent)
                                         - 1 * weight * 1/sigma2 * np.exp(exponent))
        num = np.sum(component_probs,axis=0) * np.sum(component_prob_grads2,axis=0) - np.sum(component_prob_grads,axis=0)**2
        denom = np.sum(component_probs,axis=0) ** 2
        return num/denom

    # schrodinger potential for lawgd
    def v_s(self, theta):
        return (self.ln_prob_grad(theta) ** 2 + 2 * self.ln_prob_grad_two(theta)) / 4

    # ln prob, ln prob grad
    def ln_prob_ln_prob_grad(self, theta):
        return self.ln_prob(theta), self.ln_prob_grad(theta)

    def sample(self, n_samples):
        samples = np.zeros((n_samples,))
        norm_weights = [x/sum(self.weights) for x in self.weights]
        mixture_idx = np.random.choice(len(self.weights), size=n_samples, replace=True, p=self.weights)
        for i in range(n_samples):
            sample = np.random.normal(loc=self.means[mixture_idx[i]], scale=self.variances[mixture_idx[i]])
            samples[i] = sample
        return samples


# bivariate normal
class MVN:
    def __init__(self, mu, A):
        self.mu = mu
        self.A = A

    # log density
    def ln_prob(self,theta):
        if theta.shape == (2, ):
            return -0.5 * np.sum((theta - self.mu).dot(self.A)*(theta - self.mu))
        else:
            return -0.5 * np.sum((theta - self.mu).dot(self.A)*(theta - self.mu), axis=1).T

    # grad log density
    def ln_prob_grad(self, theta):
        return -1 * np.dot(theta - self.mu, self.A)

    def neg_ln_prob_grad_torch(self, theta):
        return torch.mm(theta - torch.FloatTensor(self.mu), torch.FloatTensor(self.A))

    def ln_prob_ln_prob_grad(self, theta):
        return self.ln_prob(theta), self.ln_prob_grad(theta)

    def sample(self, n_samples):
        return np.random.multivariate_normal(mean=self.mu, cov=np.linalg.inv(self.A), size=(n_samples,))


# mixture of bivariate normals
class GeneralMixtureMVN:
    def __init__(self, means, precisions, weights):
        self.means = means
        self.precisions = precisions
        self.covs = [np.linalg.inv(a) for a in precisions]
        self.weights = weights

    def ln_prob(self, theta):
        squares = list()
        for mean, precision in zip(self.means, self.precisions):
            if theta.shape == (2,):
                squares.append(-0.5 * np.sum((theta - mean).dot(precision) * (theta - mean)))
            else:
                squares.append(-0.5 * np.sum((theta - mean).dot(precision)*(theta - mean), axis=1).T)
        weighted_exp_squares = [weight*np.exp(square) for square,weight in zip(squares, self.weights)]
        if theta.shape == (2, ):
            return np.log(np.sum(weighted_exp_squares))
        else:
            return np.log(np.sum(weighted_exp_squares,axis=0)[:,None])

    def ln_prob_grad(self,theta):
        squares = list()
        for mean, precision in zip(self.means, self.precisions):
            if theta.shape == (2,):
                squares.append(-0.5 * np.sum((theta - mean).dot(precision) * (theta - mean)))
            else:
                squares.append(-0.5 * np.sum((theta - mean).dot(precision) * (theta - mean), axis=1).T)
        nums = list()
        for mean, precision, square, weight in zip(self.means, self.precisions, squares, self.weights):
            if theta.shape == (2,):
                nums.append(-weight * np.dot(theta - mean, precision) * np.exp(square))
            else:
                nums.append(-weight * np.dot(theta - mean, precision) * np.exp(square)[:, None])
        denoms = list()
        for square, weight in zip(squares, self.weights):
            denoms.append(weight * np.exp(square))
        if theta.shape == (2,):
            return np.sum(nums, axis=0)/np.sum(denoms)
        else:
            return np.sum(nums, axis=0)/(np.sum(denoms, axis=0)[:, None])

    def neg_ln_prob_grad_torch(self, theta):
        den = 0
        top = 0
        for mean, precision, weight in zip(self.means, self.precisions, self.weights):
            exp = torch.exp(-0.5 * ((theta - torch.FloatTensor(mean)) ** 2).sum(axis=1) * precision)
            den += weight * exp
            top += weight * exp[:, None] * (theta - torch.FloatTensor(mean)) * precision
        return -top / den[:, None]

    def ln_prob_grad_two(self, theta):
        squares = list() # 'component_exponents'
        for mean, precision in zip(self.means, self.precisions):
            squares.append(-0.5 * np.sum((theta - mean).dot(precision) * (theta - mean), axis=1).T)
        nums = list() # 'component prob grads'
        for mean, precision, square, weight in zip(self.means, self.precisions, squares, self.weights):
            nums.append(-weight * np.dot(theta - mean, precision) * np.exp(square)[:, None])
        denoms = list() # 'component probs'
        for square, weight in zip(squares, self.weights):
            denoms.append(weight * np.exp(square))
        trinoms = list() # 'component prob grads 2'
        for mean, precision, square, weight in zip(self.means, self.precisions, squares, self.weights):
            trinoms.append(- 1 * weight * precision * np.exp(square)[:,None,None] +
                           1 * weight * np.dot(theta-mean,precision)[:,:,None] * np.dot(theta-mean,precision)[:,None,:] * np.exp(square)[:,None,None])
        num = np.sum(denoms, axis=0)[:,None,None] * np.sum(trinoms, axis=0) - np.sum(nums,axis=0)[:,:,None] * np.sum(nums,axis=0)[:,None,:]
        denom = (np.sum(denoms, axis=0) ** 2)[:,None,None]
        return num / denom

    def v_s(self, theta):
        return (np.sum(self.ln_prob_grad(theta) ** 2, axis=1) + 2 * np.trace(self.ln_prob_grad_two(theta), axis1=1, axis2=2)) / 4

    def ln_prob_ln_prob_grad(self, theta):
        return self.ln_prob(theta), self.ln_prob_grad(theta)

    def sample(self, n_samples):
        samples = np.zeros((n_samples, 2))
        norm_weights = [a / sum(self.weights) for a in self.weights]
        mixture_idx = np.random.choice(len(self.weights), size=n_samples, replace=True, p=norm_weights)
        for i in range(n_samples):
            samp = np.random.multivariate_normal(mean=self.means[mixture_idx[i]], cov=self.covs[mixture_idx[i]])
            samples[i] = samp
        return samples


# Rosenbrock banana
class Rosenbrock:
    def __init__(self, mean, cov, a, b):
        self.mean = mean
        self.cov = cov
        self.a = a
        self.b = b
        self.normal = MVN(self.mean, np.linalg.inv(self.cov)) # invert covariance as MVN uses precision

    def theta_transform(self, theta):
        theta_tmp = np.copy(theta)
        if theta.shape == (2,):
            theta_tmp[0] = theta[0] / self.a
            theta_tmp[1] = theta[1] * self.a + self.a * self.b * (theta[0] ** 2 + self.a ** 2)
        else:
            theta_tmp[:, 0] = theta[:, 0] / self.a
            theta_tmp[:, 1] = theta[:, 1] * self.a + self.a * self.b * (theta[:, 0] ** 2 + self.a ** 2)
        return theta_tmp

    def ln_prob(self, theta):
        theta_tmp = self.theta_transform(theta)
        return self.normal.ln_prob(theta_tmp)

    def ln_prob_grad(self, theta):
        theta_tmp = self.theta_transform(theta)
        grad_tmp = self.normal.ln_prob_grad(theta_tmp)
        if theta.shape == (2,):
            return np.array([grad_tmp[0] / self.a + grad_tmp[1] * self.a * self.b * 2 * theta[0], grad_tmp[1] * self.a])
        else:
            return np.column_stack([grad_tmp[:, 0] / self.a + grad_tmp[:, 1] * self.a * self.b * 2 * theta[:, 0],
                                    grad_tmp[:, 1] * self.a])

    def ln_prob_ln_prob_grad(self, theta):
        return self.ln_prob(theta), self.ln_prob_grad(theta)


# donut
class Donut:
    def __init__(self, radius, sigma2):
        self.radius = radius
        self.sigma2 = sigma2

    def ln_prob(self, theta):
        if theta.shape == (2,):
            r = np.sqrt(np.sum(theta**2))
            return - ((r - self.radius) ** 2 / self.sigma2)
        else:
            r = np.sqrt(np.einsum('ij,ij->i', theta, theta))
            return - ((r - self.radius) ** 2 / self.sigma2)[:,None]

    def ln_prob_grad(self, theta):
        if theta.shape == (2,):
            r = np.sqrt(np.sum(theta**2))
            return theta * ((self.radius / r - 1) * 2 / self.sigma2)
        else:
            r = np.sqrt(np.einsum('ij,ij->i', theta, theta))
            return theta * ((self.radius/r - 1) * 2 / self.sigma2)[:,None]

    def ln_prob_ln_prob_grad(self, theta):
        return self.ln_prob(theta), self.ln_prob_grad(theta)


# funnel
class Funnel:
    def __init__(self,m0, m1, s0):
        self.m0 = m0
        self.m1 = m1
        self.s0 = s0

    def f(self, theta, m, s):
        return -0.5 * np.log(2 * np.pi) - np.log(s) - 0.5 * ((theta - m) / s) ** 2

    def df_dx(self, theta, m, s):
        return - (theta - m)/(s ** 2)

    def df_ds(self, theta, m, s):
        return ((theta - m) ** 2 - s ** 2)/(s ** 3)

    def theta_transform(self, theta):
        theta_tmp = np.copy(theta)
        if theta.shape == (2,):
            theta_tmp[0] = theta[1] #- 2
            theta_tmp[1] = theta[0]
        else:
            theta_tmp[:, 0] = theta[:, 1] #- 2
            theta_tmp[:, 1] = theta[:, 0]
        return theta_tmp

    def ln_prob(self, theta):
        theta_tmp = self.theta_transform(theta)
        if theta.shape == (2,):
            s1 = np.exp(theta_tmp[0] / 2)
            return self.f(theta_tmp[0], self.m0, self.s0) + self.f(theta_tmp[1], self.m1, s1)
        else:
            s1 = np.exp(theta_tmp[:, 0]/2)
            return self.f(theta_tmp[:, 0], self.m0, self.s0) + self.f(theta_tmp[:, 1], self.m1, s1)

    def ln_prob_grad(self, theta):
        theta_tmp = self.theta_transform(theta)
        if theta.shape == (2,):
            s1 = np.exp(theta_tmp[0] / 2)
            return np.array([self.df_dx(theta_tmp[1], self.m1, s1),
                                    self.df_ds(theta_tmp[0], self.m0, self.s0) +
                                    0.5 * s1 * self.df_ds(theta_tmp[1], self.m1, s1)])
        else:
            s1 = np.exp(theta_tmp[:, 0] / 2)
            return np.column_stack([self.df_dx(theta_tmp[:, 1], self.m1, s1),
                                    self.df_ds(theta_tmp[:, 0], self.m0, self.s0) +
                                    0.5 * s1 * self.df_ds(theta_tmp[:, 1], self.m1, s1)])

    def ln_prob_ln_prob_grad(self, theta):
        return self.ln_prob(theta), self.ln_prob_grad(theta)

    def sample(self, n_samples):
        theta2 = np.random.normal(self.m0, self.s0, n_samples)
        theta1 = np.random.normal(loc=self.m1, scale=np.exp(theta2/2))
        return np.column_stack([theta1, theta2])


# squiggle
class Squiggle:
    def __init__(self, mean, cov, freq):
        self.mean = mean
        self.cov = cov
        self.freq = freq
        self.normal = MVN(self.mean, np.linalg.inv(self.cov)) # invert covariance as MVN uses precision

    def theta_transform(self, theta):
        theta_tmp = np.copy(theta)
        if theta.shape == (2,):
            theta_tmp[0] = theta[0]
            theta_tmp[1] = theta[1] + np.sin(self.freq * theta[0])
        else:
            theta_tmp[:, 0] = theta[:, 0]
            theta_tmp[:, 1] = theta[:, 1] + np.sin(self.freq * theta[:, 0])
        return theta_tmp

    def ln_prob(self, theta):
        theta_tmp = self.theta_transform(theta)
        return self.normal.ln_prob(theta_tmp)

    def ln_prob_grad(self, theta):
        theta_tmp = self.theta_transform(theta)
        grad_tmp = self.normal.ln_prob_grad(theta_tmp)
        if theta.shape == (2,):
            return np.array([grad_tmp[0] + grad_tmp[1] * self.freq * np.cos(self.freq * theta[0]),grad_tmp[1]])
        else:
            return np.column_stack([grad_tmp[:, 0] + grad_tmp[:, 1] * self.freq * np.cos(self.freq * theta[:, 0]),
                                   grad_tmp[:, 1]])

    def ln_prob_ln_prob_grad(self, theta):
        return self.ln_prob(theta), self.ln_prob_grad(theta)





