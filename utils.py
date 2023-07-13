import os
import sys
import pandas as pd
from datetime import date
from math import pi
from matplotlib import pyplot as plt
from shutil import copyfile
from sumolib import checkBinary


def create_folder(folders_name, alg):
    today = date.today()
    dirs = []
    folders_path = os.path.join(os.getcwd(), folders_name)
    if not os.path.exists(folders_path):
        os.mkdir(folders_path)
    # print(os.listdir(folders_path))
    for n in os.listdir(folders_path):
        d, index = n.split('_')
        if d == f'{alg}-{str(today)}':
            dirs.append(index)
    dirs = sorted(int(i) for i in dirs)
    if dirs:
        new_dir = dirs[-1] + 1
    else:
        new_dir = 1
    folder_path = os.path.join(folders_path, f'{alg}-{str(today)}_{str(new_dir)}')
    os.mkdir(folder_path)
    return folder_path


def create_model_folder():
    models_folder = 'models'
    models_path = os.path.join(os.getcwd(), models_folder)
    if not os.path.exists(models_path):
        os.mkdir(models_folder)

    dirs = sorted(int(i) for i in os.listdir(models_path))
    if dirs:
        new_dir = dirs[-1] + 1
    else:
        new_dir = 1

    model_path = os.path.join(models_path, str(new_dir))
    os.mkdir(model_path)
    return model_path


def create_result_folder(folder_path):
    result_path = os.path.join(os.getcwd(), folder_path)
    if not os.path.exists(result_path):
        os.makedirs(folder_path)


def plot_data(data, y_label, train_or_test, x_label='Episode'):
    plt.rcParams.update({'font.size': 15})
    plt.plot(data)
    plt.title(x_label + ' vs ' + y_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(min(data), max(data))
    fig_name = f'{train_or_test}_{y_label}.png'
    plt.savefig(fig_name, dpi=100, bbox_inches='tight')
    # plt.show()


def save_data(file_tosave, model_path):
    print('Saving data at: %s\n' % model_path)
    copyfile(file_tosave, os.path.join(model_path, file_tosave))


def set_sumo(gui=False, sumocfg_path='data/Eastway-Central.sumocfg', random=True, log_path=None, seed=-1):
    # we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # cmd mode or visual mode
    if gui:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')

    # setting the cmd to run sumo
    if random and not log_path:
        if seed < 0:
            sumo_cmd = [sumoBinary, '-c', sumocfg_path, '--random', '--no-warnings', '--no-step-log']
        else:
            sumo_cmd = [sumoBinary, '-c', sumocfg_path, "--seed", "%d" % seed, '--no-warnings', '--no-step-log']
    elif random and log_path:
        if seed < 0:
            sumo_cmd = [sumoBinary, '-c', sumocfg_path, '--random', '--no-warnings', '--no-step-log',
                        '--tripinfo-output', log_path + '_tripinfo.xml']
        else:
            sumo_cmd = [sumoBinary, '-c', sumocfg_path, '--seed', '%d' % seed, '--no-warnings', '--no-step-log',
                        '--tripinfo-output', log_path + '_tripinfo.xml']
    elif not random and log_path:
        sumo_cmd = [sumoBinary, '-c', sumocfg_path, '--no-warnings', '--no-step-log',
                    '--tripinfo-output', log_path + '_tripinfo.xml']
    else:
        sumo_cmd = [sumoBinary, '-c', sumocfg_path, '--no-warnings', '--no-step-log']

    return sumo_cmd


def plot_box():
    # Load data from Excel file
    df = pd.read_excel('result-analysis.xlsx', sheet_name='box', header=0)

    # Create a box plot
    fig, ax = plt.subplots()
    df.boxplot(ax=ax)

    # Add labels and a title
    ax.set_title('Signal Controller Performance Comparison', fontsize=16, y=1.05)
    ax.set_ylabel('Average Person Delay (s)')

    # Save the plot
    plt.savefig('Box plot of average person delay.png', dpi=300)
    plt.show()


def plot_radar():
    # Load data from Excel file
    df = pd.read_excel('result-analysis.xlsx', sheet_name='radar', header=0, index_col=0)

    # Extract the directions names from the first row
    directions = list(df.columns)

    # Extract the category names from the first column
    categories = list(df.index)

    # Get the data values as a 2D list
    data = df.values.tolist()

    # Create a radar chart
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, polar=True)

    # Set the number of directions
    num_vars = len(directions)

    # Calculate the angle for each direction
    angles = [x / float(num_vars) * 2 * pi for x in range(num_vars)]
    angles += angles[:1]

    # Set the radar chart attributes
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], directions)
    ax.set_rlabel_position(0)
    plt.yticks([15, 30, 45, 60], ["15 s", "30 s", "45 s", "60 s"], color="grey", size=7)
    # plt.ylim(0, 60)
    ax.grid(linestyle='dashed')

    # Plot the data
    for i in range(len(categories)):
        values_data = data[i]
        values_data += values_data[:1]
        ax.plot(angles, values_data, linewidth=1, linestyle='solid', label=categories[i],
                marker='o', markersize=3)
        ax.fill(angles, values_data, alpha=0)

    fig.subplots_adjust(top=0.8)

    # Add a legend
    plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), fontsize='large')

    # Add a title
    plt.title("Average Vehicle Delay in Each Direction", fontsize=16, y=1.1, fontweight='bold')

    # Save the plot
    plt.savefig('Average Vehicle Delay in Each Direction.png', dpi=300)

    # Show the plot
    plt.show()


def plot_learningcurve():
    # Load the data into a pandas DataFrame
    data = pd.read_excel('result-analysis.xlsx', sheet_name='reward')

    col_names = data.columns[1:]

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 6))

    for col in col_names:
        # Calculate the mean and standard deviation for each algorithm
        y_mean = data[col].mean()
        y_std = data[col].std()

        # Plot the mean rewards for each algorithm
        ax.plot(data['Episode'], data[col], label=col, linewidth=0.5)

        # Shade the regions between the mean +/- one standard deviation
        ax.fill_between(data['Episode'], data[col] - y_std, data[col] + y_std, alpha=0.2)

    # Set the axis labels and legend
    ax.set_xlabel('Episode', fontsize=14)
    ax.set_ylabel('Reward', fontsize=14)
    ax.set_title('Learning Curve', fontsize=18, fontweight='bold')

    ax.legend()

    # Show the plot
    plt.savefig('Reward curve.png', dpi=300)
    plt.show()


def plot_curves_two():
    # Data
    data_peak = pd.read_excel('result-analysis.xlsx', sheet_name='reward-peak')
    data_offpeak = pd.read_excel('result-analysis.xlsx', sheet_name='reward-offpeak')
    data_lst = [data_peak, data_offpeak]

    # Create the figure and axis objects
    fig, [ax1, ax2] = plt.subplots(2, 1, sharex='all', gridspec_kw={'hspace': 0.2}, figsize=(10, 6))
    ax_lst = [ax1, ax2]
    chart_names = ['Peak', 'Off-Peak']
    font = {
        # 'family': 'times new roman',
        'color': 'black',
        # 'weight': 'bold',
        'size': 12,
    }

    for i in range(2):
        data = data_lst[i]
        dqn_x = data.columns[0]
        dqn_y = data.columns[1]
        ppo_x = data.columns[2]
        ppo_names = data.columns[3:]

        # Plotting
        ax = ax_lst[i]
        ax.plot(data[dqn_x], data[dqn_y], label=dqn_y, linewidth=1)
        for ppo_name in ppo_names:
            ax.plot(data[ppo_x], data[ppo_name], label=ppo_name, linewidth=1)

        # Set the axis labels and legend
        ax.set_ylabel('Mean episode Reward', fontdict=font)
        ax.set_title(chart_names[i], fontdict=font)
        ax.grid(linestyle='dashed')

        # Add legend
        ax.legend()

    plt.xlabel('Training Step', fontdict=font)

    # Save the figure
    plt.savefig('Reward curve.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_bar():
    # Import data
    df = pd.read_excel('result-analysis.xlsx', sheet_name='turning', header=0, index_col=0)

    n_controllers = 6

    # Create the figure and other objects
    fig, [ax1, ax2] = plt.subplots(2, 1, sharex='none', gridspec_kw={'hspace': 0.2}, figsize=(10, 8))
    font = {
        # 'family': 'times new roman',
        'color': 'black',
        # 'weight': 'bold',
        'size': 12,
    }
    ax_lst = [ax1, ax2]
    df_lst = ['df1', 'df2']
    chart_names = ['Peak', 'Off-Peak']

    # Plot the bar chart for each scenario
    for i in range(2):
        ax = ax_lst[i]
        df_lst[i] = df[i * n_controllers: (i + 1) * n_controllers].T
        df_lst[i].plot(ax=ax, kind='bar')

        # Set the axis labels and legend
        ax.tick_params(axis='x', rotation=0)
        ax.set_ylabel('Average Delay (s)', fontdict=font)
        ax.set_title(chart_names[i], fontdict=font)
        ax.set_axisbelow(True)  # Set the axis below the graph element
        ax.yaxis.grid(linestyle='dashed')
        ax.legend(loc='upper right', ncol=4, fontsize=8)

    # Save the figure
    plt.savefig('Average Vehicle Delay in Each Direction (Bar)', dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()


if __name__ == '__main__':
    # plot_box()
    plot_radar()
    # create_folder(folders_name='logs', alg='DQN')
    # plot_curves_two()