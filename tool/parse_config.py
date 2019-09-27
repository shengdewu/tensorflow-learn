import xml.dom.minidom as minidom

class parse_config(object):
    def __init__(self):
        return

    @staticmethod
    def get_config(xml_path):
        dom_tree = minidom.parse(xml_path)
        collection = dom_tree.documentElement
        if collection.nodeName != 'config':
            raise RuntimeError('this is invalid nn config: the must has header "config"')

        config = dict()

        input_layer = collection.getElementsByTagName('input_layer')
        config['input_num'] = int(input_layer[0].getElementsByTagName('input_num')[0].firstChild.data)

        output_layer = collection.getElementsByTagName('output_layer')
        config['output_num'] = int(output_layer[0].getElementsByTagName('output_num')[0].firstChild.data)

        cell_layer = collection.getElementsByTagName('cell_layer')
        config['time_step'] = int(cell_layer[0].getElementsByTagName('time_step')[0].firstChild.data)
        config['cell_unit'] = [int(n.strip()) for n in cell_layer[0].getElementsByTagName('cell_unit')[0].firstChild.data.split(',')]

        optimize = collection.getElementsByTagName('optimize')
        config['batch_size'] = int(optimize[0].getElementsByTagName('batch_size')[0].firstChild.data)
        config['learn_rate'] = float(optimize[0].getElementsByTagName('learn_rate')[0].firstChild.data)
        config['decay_rate'] = float(optimize[0].getElementsByTagName('decay_rate')[0].firstChild.data)
        config['moving_decay'] = float(optimize[0].getElementsByTagName('moving_decay')[0].firstChild.data)
        config['regularize_rate'] = float(optimize[0].getElementsByTagName('regularize_rate')[0].firstChild.data)
        config['max_iter_times'] = int(optimize[0].getElementsByTagName('max_iter_times')[0].firstChild.data)

        mode = collection.getElementsByTagName('mode')
        config['mode_path'] = mode[0].getElementsByTagName('mode_path')[0].firstChild.data
        config['update_mode_freq'] = int(mode[0].getElementsByTagName('update_mode_freq')[0].firstChild.data)
        return config


