import tensorflow as tf
import gp4gw
import tensorflow_probability as tfp
import bilby
import inspect

float_type = tf.float64
tf_priors = True


def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)

    return decorator


class SampleLogJoint:

    def __init__(self, predict_log_joint_mean_var, analytic_priors, param_names, parameters_scaler, log_joint_scaler):
        assert list(param_names), '{} is not of type list'.format(param_names)

        
        
        self.predict_log_joint_mean_var = predict_log_joint_mean_var
        self.analytic_priors = analytic_priors
        self.param_names = param_names
        self.parameters_scaler = parameters_scaler
        self.log_joint_scaler = log_joint_scaler

        if any(key == 'dec' for key in param_names):
            self.bijector_list = [tfp.bijectors.Softplus()] * (len(param_names) - 1)
            self.bijector_list.insert(param_names.index('dec'), tfp.bijectors.Identity())
        if any(key == 'zenith' for key in param_names):
            self.bijector_list = [tfp.bijectors.Softplus()] * (len(param_names) - 1)
            self.bijector_list.insert(param_names.index('zenith'), tfp.bijectors.Identity())
        else:
            self.bijector_list = [tfp.bijectors.Softplus()] * len(param_names)

    def satisfy_constraints(self, x_scaled):
        """
        Function checks if analytic_priors contains any bilby Constraint objects
        and if so it evaluates them at sampled location
        :param x_scaled: this is one point in Q-dimensions
        :return: Boolean, True is constraint are satisfied
        """
        for i, key in enumerate(list(self.analytic_priors.keys())):
            if isinstance(self.analytic_priors[key], (bilby.gw.prior.Constraint, gp4gw.bbh_example.tf_Constraint)):
                chirp_mass_index = self.param_names.index('chirp_mass')
                mass_ratio_index = self.param_names.index('mass_ratio')
                if key == 'mass_ratio':
                    mass_1_index = self.param_names.index('mass_1')
                    mass_2_index = self.param_names.index('mass_2')
                    return self.analytic_priors['mass_ratio'].prob(
                        x_scaled[:, mass_2_index] / x_scaled[:, mass_1_index])
                if key =='mass_1':
                    total_mass = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(
                        x_scaled[:, chirp_mass_index], x_scaled[:, mass_ratio_index])
                    mass_1 = bilby.gw.conversion.total_mass_and_mass_ratio_to_component_masses(
                    x_scaled[:, mass_ratio_index], total_mass)[0]
                    return self.analytic_priors['mass_1'].prob(mass_1)
                if key =='mass_2':
                    total_mass = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(
                        x_scaled[:, chirp_mass_index], x_scaled[:, mass_ratio_index])
                    mass_2 = bilby.gw.conversion.total_mass_and_mass_ratio_to_component_masses(
                    x_scaled[:, mass_ratio_index], total_mass)[1]
                    return self.analytic_priors['mass_2'].prob(mass_2)
                if key =='total_mass':
                    total_mass = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(
                        x_scaled[:, chirp_mass_index], x_scaled[:, mass_ratio_index])
                    return self.analytic_priors['total_mass'].prob(total_mass) 
                else:
                    print(f'Constraint for {key} not implemented')
                    pass
            else:
                return tf.convert_to_tensor([True] * x_scaled.shape[0])

    def log_prior(self, x_scaled):
        """
        https://git.ligo.org/lscsoft/bilby/blob/master/bilby/core/prior/dict.py#L413
        :param x_scaled: this is one point in Q-dimensions- scaled to be between 0 and 1 (for ease of sampling)
        :return: sum of log priors evaluated for each parameter
        (possible because these are all independent!)
        """
        x = scale_inverse_transform(self.parameters_scaler, x_scaled)
        x = tf.cast(x, tf.float64)
        ln_prior = {}
        for key in self.param_names:
            if isinstance(self.analytic_priors[key], (bilby.gw.prior.Constraint, gp4gw.bbh_example.tf_Constraint)):
                raise NameError(f"The prior for {key} is a Constraint distribution.")
            else:
                if tf_priors == True:
                    ln_prior[key] = self.analytic_priors[key].log_prob(x[:, self.param_names.index(key)])
                else:
                    ln_prior[key] = self.analytic_priors[key].ln_prob(x[:, self.param_names.index(key)])

        ln_prior_sum = tf.math.reduce_sum(tf.convert_to_tensor(list(ln_prior.values()), dtype=float_type),
                                          axis=0)

        pass_constraint = self.satisfy_constraints(x)
        out_of_bounds = tf.ones_like(ln_prior_sum, dtype=float_type) * -float("inf")
        ln_prior_sum = tf.where(pass_constraint, ln_prior_sum, out_of_bounds)

        return ln_prior_sum

    def log_prob(self, x_scaled):
        """
        :param x_scaled: this is one point in Q-dimensions- scaled to be between 0 and 1 (for ease of sampling)
        :return: log_likelihood(x) + analytic log_prior(x)
        """
        predictions = self.predict_log_joint_mean_var(x_scaled)
#         random_jitter = tfp.distributions.Normal(loc=0, scale = 10**-3).sample([predictions[0].shape[0], 1])
        mean = scale_inverse_transform(self.log_joint_scaler, predictions[0]) #+ tf.cast(random_jitter, float_type)
#         mean = scale_inverse_transform(self.log_joint_scaler, predictions) # for gpytorch
        tensor_log_prior = self.log_prior(x_scaled)
        return tf.reshape(mean, shape=tensor_log_prior.shape) + tensor_log_prior

    @conditional_decorator(dec=tf.function(experimental_compile=True), condition=tf_priors)
    def run_chain(self, num_results, num_burnin_steps, initialise_chains, num_chains, sampler):
        """
        Run sampler over likelihood surface and evaluate prior. This returns the posterior samples
        (log_likelihood + log_prior).
        :param num_results: final number of posterior samples
        :param num_burnin_steps: number of burn-ins
        :param initialise_chains: initial sampling point for all chains. This has size (Q, 1).
        :param num_chains: min 2 chains
        :param sampler: Default is MCMC, alternative option is HMC
        """
        assert num_chains > 1, 'num_chains is 1, at least two chains must be used when sampling'
        p0 = tf.convert_to_tensor(initialise_chains, dtype=float_type)

        if sampler == 'MCMC':
            inner_sampler = tfp.mcmc.RandomWalkMetropolis(
                target_log_prob_fn=self.log_prob)

            sampler = tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=inner_sampler,
                bijector=self.bijector_list)

        if sampler == 'HMC':
            inner_sampler = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.log_prob,
                step_size=0.00052,
                num_leapfrog_steps=5)

            transitioned_kernel = tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=inner_sampler,
                bijector=self.bijector_list)

            sampler = tfp.mcmc.SimpleStepSizeAdaptation(inner_kernel=transitioned_kernel,
                                                        num_adaptation_steps=int(0.8 * num_burnin_steps))

        if sampler == 'NUTS':
            sampler = tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn=self.log_prob,
                step_size=0.00052)
        print('Running sampler with {}'.format(str(sampler.__class__).split('.')[-1].split('\'')[0]))

        results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results=1,  # thinning of chains
            current_state=p0,
            kernel=sampler
        )

        #         trace = results.trace
        #         p_accept = tf.reduce_mean(tf.exp(tf.minimum(trace.inner_results.inner_results.log_accept_ratio, 0.0)))
        states = results.all_states
        return states  # , p_accept

def scale_inverse_transform(whatever_scaler, X):
    """
    wrapper around sklearn scaler (inverse) that doesn't involve numpy
    :param parameters_scaler: An sklearn minmax scaler
    :param X: Data we are transforming
    """
    try:
        X -= whatever_scaler.min_
        X /= whatever_scaler.scale_
    except AttributeError:
        X *= whatever_scaler.scale_
        X += whatever_scaler.mean_
    return X


