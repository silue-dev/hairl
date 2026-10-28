from plotting import plot_study

benchmarks = [
    'Acrobot-v1',
    'Pendulum-v1',
    'Ant-v4',
    'HalfCheetah-v4',
    'LunarLander-v2',
    'MountainCar-v0',
    # 'LimitHoldem-v0',
]

def main():
    for benchmark in benchmarks:
        params = ['gt_ratio_p','gt_ratio_d','noise_start','noise_final']

        for p in params:
            print(f"\n▶ Generating {benchmark} overlay for {p} ...")
            plot_study(
                benchmark=benchmark,
                param_name=p
            )
        print(f"\n✔ All study plots generated under src/plots/study/{benchmark.lower()}/")

if __name__ == '__main__':
    main()
