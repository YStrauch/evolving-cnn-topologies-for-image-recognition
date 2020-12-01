import os
import json
import sys
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

if not os.path.isdir('visualise'):
    os.chdir('..')
    sys.path.insert(0, '.')
    import cga
else:
    sys.path.insert(0, '.')
    import cga


def experiment_to_yaml(data, experiment):
    f_in = 'experiments/' + experiment + '/algorithm.p'
    f_out = 'visualise/frontend/data/' + experiment + '.yaml'
    if os.path.isfile(f_out):
        return

    print(experiment)
    # set a fake GPU so the algorithm crashes in case it tries to evaluate anything
    cga.HardwareManager.set_devices(number_of_cpus=0, gpus=['fake'])
    cga.DiagramExporter().pickle_to_yaml(data, f_in, f_out)


cifar10 = cga.CIFAR10()
mnist = cga.MNIST()

experiments = [
    (cifar10,
     'data-CIFAR10__popsize-20__crossover-0.9__mutation-0.2__punishment-per-hour-0__learning-0.1_0.9_[1, 26, 43]_0.9__epochfn-const_epochs_60'
     ),
    (cifar10,
        'approach-2__data-CIFAR10__popsize-20__crossover-0.9__mutation-0.2__punishment-per-hour-0__learning-0.1_0.9_[1, 26, 43]_0.9__epochfn-const_epochs_60'
     ),
    (cifar10,
        'approach-3__data-CIFAR10__popsize-20__crossover-0.9__mutation-0.2__punishment-per-hour-0__learning-0.1_0.9_[1, 26, 43]_0.9__epochfn-const_epochs_60'
     ),

    (mnist,
     'data-MNIST__popsize-20__crossover-0.9__mutation-0.2__punishment-per-hour-0__learning-0.1_0.9_[1, 3, 5]_0.9__epochfn-const_epochs_6'
     ),

    (cifar10,
     'data-CIFAR10__popsize-20__crossover-0.9__mutation-0.2__punishment-per-hour-0.05__learning-0.1_0.9_[1, 26, 43]_0.9__epochfn-const_epochs_60'
     ),

     (cifar10,
     'data-CIFAR10__popsize-20__crossover-0.9__mutation-0.2__punishment-per-hour-0__learning-0.1_0.9_[1, 30, 50]_0.9__epochfn-linear_epochs_30_70'
     ),

     (cifar10,
     'data-CIFAR10__popsize-20__crossover-0.9__mutation-0.2__punishment-per-hour-0.05__learning-0.1_0.9_[1, 30, 50]_0.9__epochfn-linear_epochs_30_70'
     ),

    # (cifar10,
    #  'data-CIFAR10__popsize-20__crossover-0.9__mutation-0.2__punishment-per-hour-0.1__learning-0.1_0.9_[1, 30, 50]_0.9__epochfn-linear_epochs_30_70'
    #  ),

    # (cifar10,
    #  'data-CIFAR10__popsize-20__crossover-0.9__mutation-0.2__punishment-per-hour-0.05__learning-0.1_0.9_[1, 26, 43]_0.9__epochfn-epoch_evolve'
    #  ),

    # (cifar10,
    #  'data-CIFAR10__popsize-20__crossover-0.9__mutation-0.2__punishment-per-hour-0__learning-0.1_0.9_[1, 86, 143]_0.9__epochfn-const_epochs_200'
    #  ),

    # (cifar10,
    #  'data-CIFAR10__popsize-20__crossover-0.9__mutation-0.2__punishment-per-hour-0.05__learning-0.1_0.9_[1, 128, 214]_0.9__epochfn-epoch_var_linear_300_50'
    #  ),

    # (mnist,
    #  'data-MNIST__popsize-20__crossover-0.9__mutation-0.2__punishment-per-hour-0.05__learning-0.1_0.9_[1, 7, 11]_0.9__epochfn-epoch_evolve'
    #  ),


    # (mnist,
    #  'data-MNIST__popsize-20__crossover-0.9__mutation-0.2__punishment-per-hour-0.05__learning-0.1_0.9_[1, 7, 11]_0.9__epochfn-epoch_var_linear_15_2'
    #  ),



]

if os.path.isfile('visualise/frontend/available_data.js'):
    os.remove('visualise/frontend/available_data.js')

yaml_names = ['%s.yaml' % e for d, e in experiments]
with open('visualise/frontend/available_data.js', 'w') as file:
    file.writelines('var availableData = ' + json.dumps(yaml_names))

for dataset, exp in experiments:
    experiment_to_yaml(dataset, exp)

# os.chdir('visualise')
# server.serve()
