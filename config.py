args = {}
args['mode'] = 'train' # type=str)  #  mode = 'train' or 'test'
args['env_name'] = 'Pendulum-v0'   #  OpenAI gym environment name， BipedalWalker-v2
args['dataset'] = "proteins"   #  info="[proteins cox2 aids reddit-binary imdb-binary firstmm_db dblp qm9]")
args['tau'] = 0.005 # type=float)  #  target smoothing coefficient
args['target_update_interval'] = 1 # type=int)
args['iteration'] = 5 # type=int)

args['learning_rate'] = 3e-4 # type=float)
args['gamma'] = 0.99 # type=int)  #  discounted factor
args['capacity'] = 1000 # type=int)  #  replay buffer size ori=50000
args['num_iteration'] = 100000 # type=int)  #   num of  games
args['batch_size'] = 64 # type=int)  #  mini batch size
args['seed'] = 1 # type=int)
args['dropout'] = 0.5
# optional parameters
args['num_hidden_layers'] = 2 # type=int)
args['sample_frequency'] = 256 # type=int)
args['activation'] = 'Relu' # type=str)
args['render'] = False # type=bool)  #  show UI or not
args['log_interval'] = 50 # type=int)  #
args['load'] = '2022-05-22-16'#"2022-05-01-20" # type=bool)  #  load model

args['render_interval'] = 100 # type=int)  #  after render_interval  the env.render() will work
args['policy_noise'] = 0.1 # type=float)
args['noise_clip'] = 0.5 # type=float)
args['policy_delay'] = 2 # type=int)
# args['exploration_noise'] = 0.2 # type=float)  # ori=0.1
args['max_episode'] = 500 # type=int)
args['print_log'] = 5 # type=int)
