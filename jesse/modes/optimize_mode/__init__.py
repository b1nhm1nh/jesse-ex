import os
import traceback
from math import log10
from multiprocessing import cpu_count

import click
import numpy as np
import ray
from optuna.samplers import NSGAIISampler
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch
import shutil

import jesse.helpers as jh
import jesse.services.required_candles as required_candles
from jesse import exceptions
from jesse.config import config
from jesse.modes.backtest_mode import simulator
from jesse.routes import router
from jesse.services import metrics as stats
from jesse.services.validators import validate_routes
from jesse.store import store


class Optimizer():
    def __init__(self, training_candles, optimal_total: int, cpu_cores: int, iterations: int) -> None:
        if len(router.routes) != 1:
            raise NotImplementedError('optimize_mode mode only supports one route at the moment')

        self.strategy_name = router.routes[0].strategy_name
        self.optimal_total = optimal_total
        self.exchange = router.routes[0].exchange
        self.symbol = router.routes[0].symbol
        self.timeframe = router.routes[0].timeframe
        StrategyClass = jh.get_strategy_class(self.strategy_name)
        self.strategy_hp = StrategyClass.hyperparameters(None)
        self.solution_len = len(self.strategy_hp)
        self.iterations = iterations

        if self.solution_len == 0:
            raise exceptions.InvalidStrategy('Targeted strategy does not implement a valid hyperparameters() method.')

        if cpu_cores > cpu_count():
            raise ValueError(f'Entered cpu cores number is more than available on this machine which is {cpu_count()}')
        elif cpu_cores == 0:
            self.cpu_cores = cpu_count()
        else:
            self.cpu_cores = cpu_cores

        self.training_candles = training_candles

        key = jh.key(self.exchange, self.symbol)
        training_candles_start_date = jh.timestamp_to_time(self.training_candles[key]['candles'][0][0]).split('T')[0]
        training_candles_finish_date = jh.timestamp_to_time(self.training_candles[key]['candles'][-1][0]).split('T')[0]

        self.training_initial_candles = []

        for c in config['app']['considering_candles']:
            self.training_initial_candles.append(
                required_candles.load_required_candles(c[0], c[1], training_candles_start_date,
                                                       training_candles_finish_date))

        self.study_name = f'{self.strategy_name}-{self.exchange}-{self.symbol}-{self.timeframe}'

    def objective_function(self, search_space):
        score = np.nan

        # init candle store
        store.candles.init_storage(5000)
        # inject required TRAINING candles to the candle store

        for num, c in enumerate(config['app']['considering_candles']):
            required_candles.inject_required_candles_to_store(
                self.training_initial_candles[num],
                c[0],
                c[1]
            )
        # run backtest simulation
        simulator(self.training_candles, search_space)

        training_data = stats.trades(store.completed_trades.trades, store.app.daily_balance)
        total_effect_rate = log10(training_data['total']) / log10(self.optimal_total)
        total_effect_rate = min(total_effect_rate, 1)
        ratio_config = jh.get_config('env.optimization.ratio', 'sharpe')
        if ratio_config == 'sharpe':
            ratio = training_data['sharpe_ratio']
            ratio_normalized = jh.normalize(ratio, -.5, 5)
        elif ratio_config == 'calmar':
            ratio = training_data['calmar_ratio']
            ratio_normalized = jh.normalize(ratio, -.5, 30)
        elif ratio_config == 'sortino':
            ratio = training_data['sortino_ratio']
            ratio_normalized = jh.normalize(ratio, -.5, 15)
        elif ratio_config == 'omega':
            ratio = training_data['omega_ratio']
            ratio_normalized = jh.normalize(ratio, -.5, 5)
        elif ratio_config == 'serenity':
            ratio = training_data['serenity_index']
            ratio_normalized = jh.normalize(ratio, -.5, 15)
        elif ratio_config == 'smart sharpe':
            ratio = training_data['smart_sharpe']
            ratio_normalized = jh.normalize(ratio, -.5, 5)
        elif ratio_config == 'smart sortino':
            ratio = training_data['smart_sortino']
            ratio_normalized = jh.normalize(ratio, -.5, 15)
        else:
            raise ValueError(
                f'The entered ratio configuration `{ratio_config}` for the optimization is unknown. Choose between sharpe, calmar, sortino, serenity, smart shapre, smart sortino and omega.')

        if ratio > 0:
            score = total_effect_rate * ratio_normalized

        # reset store
        store.reset()
        tune.report(score=score)

    def get_search_space(self):
        config = {}
        for st_hp in self.strategy_hp:
            if st_hp['type'] is int:
                if 'step' not in st_hp:
                    st_hp['step'] = 1
                config[st_hp['name']] = tune.choice(range(st_hp['min'], st_hp['max'] + st_hp['step'], st_hp['step']))
            elif st_hp['type'] is float:
                if 'step' not in st_hp:
                    st_hp['step'] = 0.1
                decs = str(st_hp['step'])[::-1].find('.')
                config[st_hp['name']] = tune.choice(
                    np.trunc(np.arange(st_hp['min'], st_hp['max'] + st_hp['step'], st_hp['step']) * 10 ** decs) / (
                                10 ** decs))
            elif st_hp['type'] is bool:
                config[st_hp['name']] = tune.choice([True, False])
            else:
                raise TypeError('Only int, bool and float types are implemented')
        return config

    def run(self):

        ray.init(num_cpus=self.cpu_cores)

        search_space = self.get_search_space()

        ## todo make GA settings editable by user
        sampler = NSGAIISampler(population_size=300, mutation_prob=0.0333, crossover_prob=0.6, swapping_prob=0.05)

        local_dir = "./storage/genetics/ray"
        resume = False

        study_dir = local_dir+"/"+self.study_name
        # if temp file exists, load data to resume previous session
        if os.path.exists(study_dir) and click.confirm(
                'Previous session detected. Do you want to resume?', default=True
        ):
            resume = True
        else:
            shutil.rmtree(study_dir, ignore_errors=True)


        optuna_search = OptunaSearch(
            search_space,
            sampler=sampler,
            metric="score",
            mode="max")

        analysis = tune.run(
            self.objective_function,
            search_alg=optuna_search,
            local_dir=local_dir,
            num_samples=self.iterations,
            name=self.study_name,
            log_to_file=True,
            max_concurrent_trials=self.cpu_cores,
            resume=resume
        )

        print(f"Best config: {analysis.best_config}")


def optimize_mode_ray(start_date: str, finish_date: str, optimal_total: int, cpu_cores: int,
                      iterations: int) -> None:
    # clear the screen
    click.clear()

    # validate routes
    validate_routes(router)

    training_candles = get_training_candles(start_date, finish_date)

    optimizer = Optimizer(training_candles, optimal_total, cpu_cores, iterations)

    print('Starting optimization...')

    optimizer.run()


def get_training_candles(start_date_str: str, finish_date_str: str):
    # Load candles (first try cache, then database)
    from jesse.modes.backtest_mode import load_candles
    return load_candles(start_date_str, finish_date_str)
