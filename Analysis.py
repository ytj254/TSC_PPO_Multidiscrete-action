from xml.dom.minidom import parse
import xml.dom.minidom
import numpy as np


def analysis_cv(input_path):
    # Open xml
    domtree = xml.dom.minidom.parse(input_path)

    # Obtain file element
    collection = domtree.documentElement
    tripinfos = collection.getElementsByTagName('tripinfo')

    stats_keys = ['delay', 'cv_delay', 'c_delay',
                  'NW', 'NS', 'NE', 'EN', 'EW', 'ES', 'SE', 'SN', 'SW', 'WS', 'WE', 'WN']
    delay_stats = dict((k, []) for k in stats_keys)

    for tripinfo in tripinfos:
        depart_time = float(tripinfo.getAttribute('depart'))
        if 600 <= depart_time <= 4200:

            rou = tripinfo.getAttribute('id').split('.')[0]

            v_delay = float(tripinfo.getAttribute('timeLoss'))

            delay_stats['delay'].append(v_delay)

            vtype = tripinfo.getAttribute('vType')

            if vtype == 'cv':
                # print(v_delay)
                delay_stats['cv_delay'].append(v_delay)
            else:
                delay_stats['c_delay'].append(v_delay)

            if rou in delay_stats.keys():
                delay_stats[rou].append(v_delay)

    # Mean delay, if there are empty values, such as no cv scenarios, this code will raise 'Mean of empty slice' error
    delay_means = {k: np.mean(v) for k, v in delay_stats.items()}
    mean_list = list(v for v in delay_means.values())

    return mean_list


if __name__ == '__main__':
    print(analysis_cv('2_tripinfo.xml'))