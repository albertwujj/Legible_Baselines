import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance

# BIG PICTURE #

# Model will be given steps of experience to train its value function and policy.

# Runner will run the agent, sampling the action from the Model's policy at each step,
# calculating the value and advantage for each step's state-action pair,
# and finally returning all steps (with associated info)

# learn() will call Runner.run() once per "update",
# and train the Model on all steps of experience (separated into minibatches) from that call, for a certain # of epochs

#####

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm):
        sess = tf.get_default_session()

        # Creates a NN that outputs both action and value
        # (sharing parameters except for the final, fully-connected, layers)
        # Used only to sample actions to take at agent run-time
        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)

        # Creates the same NN (with same parameters), but to train, on multiple steps of experience at once
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)


        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        # Train_model.pd is a probability distribution (over possible actions) made from the policy's output

        # -log prob. of action given policy's output
        neglogpac = train_model.pd.neglogp(A)

        # entropy of policy output
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        # model will minimize the clipped or the non-clipped loss, whichever is greater
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # R is the sum of returns, or the value target
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Loss is negative of the objective function (from TRPO paper):
        #  ratio of prob. of action under new policy vs under curr policy, multiplied by advantage of action
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))


        # Sets loss to sum of policy loss and value loss
        # with an entropy bonus (improves exploration)
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # sets one call of sess.run() to apply the gradients minimizing the loss
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)


        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X: obs, A: actions, ADV: advs, R: returns, LR: lr,
                      CLIPRANGE: cliprange, OLDNEGLOGPAC: neglogpacs, OLDVPRED: values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)  # pylint: disable=E1101


class Runner(object):
    def __init__(self, *, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        # can train on multiple env in parallel
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        epinfos = []

        # runs agent for nsteps, keeping track of steps
        for _ in range(self.nsteps):
            # uses model to evaluate the observation, getting its value and sampling an action to take
            # self.states is only used in recurrent policies
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            # tracking neg. log prob. of having gotten the sampled action
            mb_neglogpacs.append(neglogpacs)
            # track at each step which envs are "done"
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            self.env.envs[0].render()
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)

        # convert batch of steps to batch of rollouts (that the model can train on)
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        # value of the final state (multiple final states if multiple envs)
        last_values = self.model.value(self.obs, self.states, self.dones)
        mb_returns = np.zeros_like(mb_rewards)

        # calculates (estimates for) the advantages of each step's state-action pair
        # (very similar to TD-lambda value estimate)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        # t is current step index. Working backwards thru the steps
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                # serves to multiply out the values for the envs that are done(at the next step)
                nextnonterminal = 1.0 - mb_dones[t + 1]
                # value of next step (multiple if multiple envs)
                nextvalues = mb_values[t + 1]

            # gamma is reward decay constant, lam is lambda parameter in exponentially-weighted averages
            # delta is the one-step TD error from current step
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]

            # for any n, the n-step TD error from a step is an estimate for the advantage of the step
            # but it is biased because our values are not exact
            # so we take an exponentially-weighted average of all the n-step TD errors, as our adv estimate
            # which is mathematically equivalent
            # to setting each step's advantage to it's one-step TD error plus the decayed advantage of the next step
            # (where the last step's advantage is 0):
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            # see GAE paper
          
        # for each step, its prior value plus its advantage estimate, 
        # an exponentially weighted average over all n-step TD errors,
        # is exactly the TD-lambda value estimate (by definition)
        # so mb_returns becomes the new target values for our Model's value function.
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states, epinfos)


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val

    return f


def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
          save_interval=10):

    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    # nbatch is the number of steps that will be taken in a single call to runner.run()
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                               nbatch_train=nbatch_train,
                               nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    # nbatch is the number of steps that will be taken in a single call to runner.run(),
    # which is called once an "update"
    nupdates = total_timesteps // nbatch
    for update in range(1, nupdates + 1):
        assert nbatch % nminibatches == 0

        # nbatch_train is number of steps we will put in 1 minibatch
        nbatch_train = nbatch // nminibatches
        tstart = time.time()

        # (optionally) decaying learning rate and clip range
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()  # pylint: disable=E0632
        epinfobuf.extend(epinfos)
        mblossvals = []


        if states is None:  # nonrecurrent version

            # Will not separate steps by environment, so steps from diff. environments will be mixed up,
            # as the order steps are run through the model do not matter for non-recurrent networks

            inds = np.arange(nbatch)

            # train noptepochs times on each set of steps
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                # train on all steps from current update
                # separated into nminibatches batches
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))

        else:  # recurrent version (for recurrent networks), will separate steps by environment

            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)

            # separates (indices to) the steps by environment
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)

            # envs per minibatch
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    # the randomnly selected environments for this minibatch
                    mbenvinds = envinds[start:end]
                    # all the steps of the randomly selected environments, put in one vector in order
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # log the info
        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update * nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update * nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)

    env.close()


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
