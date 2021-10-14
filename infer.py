from train_dist import *
from rich import print
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def infer_testa(cfg_file, checkpoint_file, output_file):
    data_file = 'data/final_dataset_testA.tsv'

    f = open(output_file, 'w')
    f.write('id,delta_g\n')

    cfg = yaml.load(open(cfg_file, 'r', encoding="utf-8", ).read(), Loader=yaml.FullLoader)
    print(cfg_file)
    print(cfg)

    x2_model, x2_alphabet = esm.pretrained.esm1_t6_43M_UR50S()
    x2_batch_converter = x2_alphabet.get_batch_converter()
    model = MyModel(x2_model)

    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
    model.to(device)
    model.eval()
    print("=> Resume: loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))
    print(model)

    data = open(data_file).readlines()[1:]
    data = [line.strip().split('\t')[0:1]+line.strip().split('\t')[2:5]+['0'] for line in data]
    data = np.array(data)
    dataset = BaseSequence(data, cfg, mode='infer')
    dataloader = DataLoader(
        dataset, batch_size=cfg['batch_size'], num_workers=cfg['nb_worker'], pin_memory=True)

    with torch.no_grad():
        for i, (id_batch, x1_batch, x2_batch, x3_batch, y_batch) in enumerate(dataloader):
            x1_batch = [('protein%d' % ii, x) for ii, x in enumerate(x1_batch)]
            x2_batch = [('protein%d' % ii, x) for ii, x in enumerate(x2_batch)]
            x3_batch = [('protein%d' % ii, x) for ii, x in enumerate(x3_batch)]

            _, _, x1_input_batch = x2_batch_converter(x1_batch)
            _, _, x2_input_batch = x2_batch_converter(x2_batch)
            _, _, x3_input_batch = x2_batch_converter(x3_batch)

            if torch.cuda.is_available():
                x1_input_batch = x1_input_batch.to(device)
                x2_input_batch = x2_input_batch.to(device)
                x3_input_batch = x3_input_batch.to(device)

            output_batch = model([x1_input_batch, x1_batch], [x2_input_batch, x2_batch], [x3_input_batch, x3_batch])
            output_batch = output_batch.reshape([-1, ]).double()

            loss_mode = cfg.get('loss_mode', 'mse')
            if loss_mode == 'mse':
                output_batch = output_batch * cfg['std_y'] + cfg['mean_y']
            elif loss_mode == 'bce':
                output_batch = torch.sigmoid(output_batch)
                output_batch = output_batch * (cfg['max_y']-cfg['min_y']) + cfg['min_y']
            output_batch = output_batch.reshape([-1, ])

            output_batch = output_batch.cpu().numpy()
            for id_, out_ in zip(id_batch, output_batch):
                out_line = f"{id_},{out_}"
                f.write(out_line+'\n')
    
    f.close()




if __name__ == '__main__':
    cfg_file = 'configs/v1.yaml'
    checkpoint_file = 'output/v1_mse/checkpoint_1.0280.pth.tar'
    output_file = 'submit/testa_v1.csv'
    infer_testa(cfg_file, checkpoint_file, output_file)
